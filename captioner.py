"""
Sends each segment clip to Nemotron Nano 12B v2 VL (served via vLLM) and asks
for a structured JSON caption -- not just free text. This matters a lot for
retrieval later: "how many people" / "is there a forklift" style questions
are much more reliable against structured fields than against buried prose.

Requires a running server, e.g.:
    vllm serve nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 \
        --trust-remote-code --dtype bfloat16 \
        --max-model-len 32768 --gpu-memory-utilization 0.90 \
        --allowed-local-media-path /home/claude/video_qa_pipeline \
        --served-model-name nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16
"""
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from openai import OpenAI

from config import (
    VLLM_BASE_URL, VLLM_API_KEY, MODEL_NAME,
    CAPTION_TEMPERATURE, CAPTION_MAX_TOKENS, CAPTION_SYSTEM_PROMPT, CAPTION_RETRIES,
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
    # Strip code fences if the model added them despite instructions
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


def caption_segment(client: OpenAI, segment: Segment) -> Caption:
    video_url = f"file://{Path(segment.clip_path).resolve()}"
    messages = [
        {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "video_url", "video_url": {"url": video_url}},
            ],
        },
    ]

    last_err = None
    for attempt in range(CAPTION_RETRIES + 1):
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=CAPTION_TEMPERATURE,
            max_tokens=CAPTION_MAX_TOKENS,
        )
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

    # Fell through all retries -- keep the raw text so nothing is silently lost
    return Caption(
        segment_id=segment.segment_id, video_id=segment.video_id,
        start_ts=segment.start_ts, end_ts=segment.end_ts,
        description=f"[CAPTION PARSE FAILED after {CAPTION_RETRIES + 1} attempts: {last_err}]",
        people_count=0, objects=[], actions=[], on_screen_text="", setting="",
    )


def caption_all(segments: list[Segment], work_dir: str) -> list[Caption]:
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
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