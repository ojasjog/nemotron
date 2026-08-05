from dotenv import load_dotenv
load_dotenv()
"""
Sends each segment clip to Nemotron Nano 12B v2 VL and asks for a structured
JSON caption -- not just free text.

What changed from the original:
- caption_all() now fires requests concurrently (ThreadPoolExecutor) instead
  of one at a time -- this is the dominant lever for wall-clock time, since
  each call is a network round-trip + hosted inference wait, and those calls
  have zero dependency on each other.
- If config.PRIORITY_WINDOW is set, segments overlapping it are submitted
  first, so they complete first.
- Captions are streamed out via an `on_caption` callback as each one
  completes (not just written to disk at the end), so a caller (pipeline.py)
  can index them incrementally and flush to disk periodically -- letting a
  second process run qa.py against partial results while this keeps going.
"""
import base64
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

from openai import OpenAI

from config import (
    CAPTION_BACKEND,
    VLLM_BASE_URL, VLLM_API_KEY, MODEL_NAME,
    HOSTED_BASE_URL, HOSTED_MODEL_NAME, HOSTED_API_KEY_ENV_VAR,
    CAPTION_TEMPERATURE, CAPTION_MAX_TOKENS, CAPTION_SYSTEM_PROMPT, CAPTION_RETRIES,
    CAPTION_FPS, CAPTION_WORKERS, PRIORITY_WINDOW,
)
from segment_video import Segment

CAPTION_PROMPT = """Describe exactly what happens in this video clip. Respond with ONLY a JSON object, no markdown fences, no extra text, matching this schema:

{
  "description": "1-3 sentence plain-language summary of what happens",
  "people_count": <integer, distinct people visible in this clip>,
  "objects": ["list", "of", "notable objects visible, e.g. forklift, ladder, sign"],
  "actions": ["list of distinct actions/events, e.g. 'person enters from left', 'box falls off shelf'"],
  "on_screen_text": "any readable text/signage visible, verbatim, or empty string if none",
  "setting": "brief description of the location/environment"
}

Be literal and specific. Do not guess at things you cannot see. If nothing notable happens, say so plainly in "description" and leave lists empty."""

HOSTED_INLINE_SIZE_WARNING_BYTES = 15 * 1024 * 1024  # ~15MB raw file, generous margin


@dataclass
class Caption:
    segment_id: str
    video_id: str
    start_ts: float
    end_ts: float
    description: str
    people_count: int
    objects: list
    actions: list
    on_screen_text: str
    setting: str


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


def build_client() -> OpenAI:
    if CAPTION_BACKEND == "hosted":
        api_key = os.environ.get(HOSTED_API_KEY_ENV_VAR)
        if not api_key:
            raise RuntimeError(
                f"CAPTION_BACKEND is 'hosted' but ${HOSTED_API_KEY_ENV_VAR} is not set. "
                f"Get a free key at https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl "
                f"then: import os; os.environ['{HOSTED_API_KEY_ENV_VAR}'] = 'nvapi-...'"
            )
        return OpenAI(base_url=HOSTED_BASE_URL, api_key=api_key)
    return OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


def _video_content_block(clip_path: str) -> dict:
    if CAPTION_BACKEND == "hosted":
        size = Path(clip_path).stat().st_size
        if size > HOSTED_INLINE_SIZE_WARNING_BYTES:
            print(
                f"  [warn] {clip_path} is {size / 1e6:.1f}MB -- hosted API may reject "
                f"large inline payloads. Consider shorter/lower-res segments if this fails."
            )
        b64 = base64.b64encode(Path(clip_path).read_bytes()).decode("utf-8")
        return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}}
    else:
        video_url = f"file://{Path(clip_path).resolve()}"
        return {"type": "video_url", "video_url": {"url": video_url}}


def caption_segment(client: OpenAI, segment: Segment) -> Caption:
    messages = [
        {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                _video_content_block(segment.clip_path),
            ],
        },
    ]

    model_name = HOSTED_MODEL_NAME if CAPTION_BACKEND == "hosted" else MODEL_NAME
    extra_body = {"media_io_kwargs": {"video": {"fps": CAPTION_FPS}}} if CAPTION_BACKEND == "local" else None

    last_err = None
    for attempt in range(CAPTION_RETRIES + 1):
        try:
            if extra_body:
                resp = client.chat.completions.create(
                    model=model_name, messages=messages,
                    temperature=CAPTION_TEMPERATURE, max_tokens=CAPTION_MAX_TOKENS,
                    extra_body=extra_body,
                )
            else:
                resp = client.chat.completions.create(
                    model=model_name, messages=messages,
                    temperature=CAPTION_TEMPERATURE, max_tokens=CAPTION_MAX_TOKENS,
                )
        except Exception as e:
            last_err = e
            continue

        raw: str = resp.choices[0].message.content or ""
        if not raw:
            last_err = ValueError("model returned empty content (no text in response)")
            continue
        try:
            data = _extract_json(raw)
            return Caption(
                segment_id=segment.segment_id, video_id=segment.video_id,
                start_ts=segment.start_ts, end_ts=segment.end_ts,
                description=data.get("description", ""),
                people_count=int(data.get("people_count", 0) or 0),
                objects=data.get("objects", []) or [],
                actions=data.get("actions", []) or [],
                on_screen_text=data.get("on_screen_text", "") or "",
                setting=data.get("setting", "") or "",
            )
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            continue

    return Caption(
        segment_id=segment.segment_id, video_id=segment.video_id,
        start_ts=segment.start_ts, end_ts=segment.end_ts,
        description=f"[CAPTION FAILED after {CAPTION_RETRIES + 1} attempts: {last_err}]",
        people_count=0, objects=[], actions=[], on_screen_text="", setting="",
    )


def _overlaps_priority(seg: Segment) -> bool:
    if PRIORITY_WINDOW is None:
        return False
    p_start, p_end = PRIORITY_WINDOW
    return seg.start_ts < p_end and seg.end_ts > p_start


def _ordered(segments: list[Segment]) -> list[Segment]:
    priority = [s for s in segments if _overlaps_priority(s)]
    rest = [s for s in segments if not _overlaps_priority(s)]
    return priority + rest


def caption_all(segments: list[Segment], work_dir: str, on_caption=None) -> list[Caption]:
    """
    on_caption(caption: Caption, index_in_file_order: int) is called from a
    worker thread the moment each caption completes -- use it to index
    incrementally (see pipeline.py). If you don't pass it, behavior is the
    same as before: everything runs, then captions.jsonl is written once at
    the end, ordered by file position.
    """
    client = build_client()
    id_to_pos = {seg.segment_id: i for i, seg in enumerate(segments)}
    captions: list[Caption] = [None] * len(segments)
    lock = threading.Lock()

    submit_order = _ordered(segments)
    if PRIORITY_WINDOW is not None:
        n_priority = sum(1 for s in submit_order if _overlaps_priority(s))
        print(f"      priority window {PRIORITY_WINDOW} -> {n_priority} segments captioned first")

    out_path = Path(work_dir) / segments[0].video_id / "captions.jsonl"

    done = 0
    with ThreadPoolExecutor(max_workers=CAPTION_WORKERS) as ex:
        futures = {ex.submit(caption_segment, client, seg): seg for seg in submit_order}
        for fut in as_completed(futures):
            seg = futures[fut]
            cap = fut.result()
            pos = id_to_pos[seg.segment_id]
            with lock:
                captions[pos] = cap
                done += 1
                # append-as-you-go so captions.jsonl is always a valid partial
                # record on disk, not just written once at the very end
                with open(out_path, "a") as f:
                    f.write(json.dumps(asdict(cap)) + "\n")
            print(f"  [{done}/{len(segments)}] [{seg.segment_id}] {cap.description[:80]}")
            if on_caption is not None:
                on_caption(cap, pos)

    return captions


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python captioner.py <video_id>  (run segment_video.py first)")
        sys.exit(1)
    from config import WORK_DIR
    video_id = sys.argv[1]
    manifest = json.loads((Path(WORK_DIR) / video_id / "segments.json").read_text())
    segments = [Segment(**m) for m in manifest]
    # truncate captions.jsonl since caption_all now appends, not overwrites
    (Path(WORK_DIR) / video_id / "captions.jsonl").write_text("")
    caption_all(segments, WORK_DIR)