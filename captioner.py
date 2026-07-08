from dotenv import load_dotenv
load_dotenv()
"""
Sends each segment clip to Nemotron Nano 12B v2 VL and asks for a structured
JSON caption -- not just free text. This matters a lot for retrieval later:
"how many people" / "is there a forklift" style questions are much more
reliable against structured fields than against buried prose.

Supports two backends (config.CAPTION_BACKEND):
  "local"  -- your own vLLM server (needs GPU hardware Nemotron VL actually
              supports: L40S/A100/H100/B200-class -- NOT T4/Colab-free-tier).
  "hosted" -- NVIDIA's hosted API (build.nvidia.com). All GPU work happens on
              NVIDIA's infrastructure, so this is the right choice on Colab's
              T4 or any machine that can't run the model locally.
              Get a free key at https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl
              then: import os; os.environ["NVIDIA_API_KEY"] = "nvapi-..."

Local mode sends clips as file:// paths (vLLM reads them off disk directly).
Hosted mode sends clips as base64 data URIs (a remote API can't read your
local filesystem).
"""
import base64
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from openai import OpenAI

from config import (
    CAPTION_BACKEND,
    VLLM_BASE_URL, VLLM_API_KEY, MODEL_NAME,
    HOSTED_BASE_URL, HOSTED_MODEL_NAME, HOSTED_API_KEY_ENV_VAR,
    CAPTION_TEMPERATURE, CAPTION_MAX_TOKENS, CAPTION_SYSTEM_PROMPT, CAPTION_RETRIES,
    CAPTION_FPS,
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

# NVIDIA-hosted VLM endpoints have historically rejected inline base64 media
# above roughly this size (larger media needs their separate assets-upload
# API). Re-encode/downscale your clips if you hit this.
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

    # The fps override is a vLLM-specific extra_body field -- reliable when
    # self-hosting, but not guaranteed to be honored (or even accepted) by
    # the hosted API, so only send it in local mode.
    extra_body = {"media_io_kwargs": {"video": {"fps": CAPTION_FPS}}} if CAPTION_BACKEND == "local" else None

    last_err = None
    for attempt in range(CAPTION_RETRIES + 1):
        try:
            if extra_body:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=CAPTION_TEMPERATURE,
                    max_tokens=CAPTION_MAX_TOKENS,
                    extra_body=extra_body,
                )
            else:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=CAPTION_TEMPERATURE,
                    max_tokens=CAPTION_MAX_TOKENS,
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
                segment_id=segment.segment_id,
                video_id=segment.video_id,
                start_ts=segment.start_ts,
                end_ts=segment.end_ts,
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


def caption_all(segments: list[Segment], work_dir: str) -> list[Caption]:
    client = build_client()
    captions = []
    for seg in segments:
        cap = caption_segment(client, seg)
        captions.append(cap)
        print(f"  [{seg.segment_id}] {cap.description[:80]}")

    out_path = Path(work_dir) / segments[0].video_id / "captions.jsonl"
    with open(out_path, "w") as f:
        for c in captions:
            f.write(json.dumps(asdict(c)) + "\n")
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
    caption_all(segments, WORK_DIR)