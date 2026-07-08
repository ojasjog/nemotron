"""
Answers a question about a video using its FAISS index.

Two paths, chosen by simple intent detection:

- "count_all" -- questions like "how many people appear in total" need a
  full scan across every segment, not top-k retrieval. Vector search can
  miss segments that aren't semantically close to the question wording even
  though they contain the relevant count, and it has no way to aggregate
  across segments anyway. NOTE: this reports the max people_count seen in
  any single segment as a conservative "at least" estimate -- there's no
  cross-segment person identity tracking here, so summing counts would
  double-count anyone visible in more than one segment. Worth flagging to
  the user in the UI, not just papering over it.
- everything else -- semantic search over the FAISS index (wider k for
  existence-style questions like "is there a forklift", since those need to
  rule out ~all segments, not just the closest match), then the retrieved
  captions are handed to the LLM to write a direct answer with timestamp
  citations.
"""
import json
import re
from pathlib import Path
from typing import Any

from config import TOP_K, TOP_K_WIDE, QA_TEMPERATURE, QA_MAX_TOKENS, CAPTION_BACKEND, MODEL_NAME, HOSTED_MODEL_NAME
from captioner import build_client
from embedder import search

COUNT_ALL_PATTERN = re.compile(r"how many (people|person|persons|individuals|workers)\b.*\b(total|in total|altogether|overall|appear)\b|total number of people", re.I)
EXISTENCE_PATTERN = re.compile(r"^\s*(is|are|was|were|does|do|did)\s+(there|any|it)\b", re.I)

ANSWER_SYSTEM_PROMPT = """You answer questions about a video using ONLY the timestamped clip descriptions provided. Rules:
- Base your answer strictly on the provided clips; never invent details not present in them.
- If the clips don't contain enough information to answer, say so plainly rather than guessing.
- When you reference something that happens, cite its timestamp in MM:SS format, e.g. "at 00:42".
- Be direct and concise -- a sentence or two is usually enough."""


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _format_clips_for_prompt(clips: list[dict]) -> str:
    lines = []
    for c in clips:
        lines.append(
            f"[{format_time(c['start_ts'])}-{format_time(c['end_ts'])}] {c['description']} "
            f"(people: {c['people_count']}, objects: {', '.join(c['objects']) or 'none'}, "
            f"actions: {', '.join(c['actions']) or 'none'}"
            + (f", text visible: \"{c['on_screen_text']}\"" if c.get("on_screen_text") else "")
            + ")"
        )
    return "\n".join(lines)


def _all_captions(video_id: str, work_dir: str) -> list[dict]:
    path = Path(work_dir) / video_id / "index_meta.json"
    return json.loads(path.read_text())


def answer_count_all_people(video_id: str, work_dir: str, question: str) -> dict:
    captions = _all_captions(video_id, work_dir)
    if not captions:
        return {"answer": "This video hasn't been processed yet.", "citations": []}

    best = max(captions, key=lambda c: c.get("people_count", 0))
    max_count = best.get("people_count", 0)

    answer = (
        f"The most people visible at once in any single clip is {max_count}, "
        f"at {format_time(best['start_ts'])}-{format_time(best['end_ts'])}. "
        f"Note: this pipeline doesn't track individual identities across clips, so this is "
        f"a conservative \"at least\" estimate from the busiest single moment, not a guaranteed "
        f"total count of unique people across the whole video."
    )
    return {
        "answer": answer,
        "citations": [{"segment_id": best["segment_id"], "start_ts": best["start_ts"], "end_ts": best["end_ts"]}],
    }


def answer_question(video_id: str, work_dir: str, question: str) -> dict:
    """Main entry point. Returns {"answer": str, "citations": [{"segment_id", "start_ts", "end_ts"}, ...]}"""
    if COUNT_ALL_PATTERN.search(question):
        return answer_count_all_people(video_id, work_dir, question)

    k = TOP_K_WIDE if EXISTENCE_PATTERN.search(question) else TOP_K
    clips = search(video_id, work_dir, question, k=k)

    if not clips:
        return {"answer": "This video hasn't been processed yet.", "citations": []}

    context = _format_clips_for_prompt(clips)
    messages: list[Any] = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Clips from the video:\n{context}\n\nQuestion: {question}"},
    ]

    client = build_client()
    model_name = HOSTED_MODEL_NAME if CAPTION_BACKEND == "hosted" else MODEL_NAME
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=QA_TEMPERATURE,
        max_tokens=QA_MAX_TOKENS,
    )
    answer_text = resp.choices[0].message.content or "I couldn't generate an answer -- please try rephrasing."

    citations = [
        {"segment_id": c["segment_id"], "start_ts": c["start_ts"], "end_ts": c["end_ts"], "relevance": c["score"]}
        for c in clips
    ]
    return {"answer": answer_text, "citations": citations}


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print('Usage: python qa.py <video_id> "your question"')
        sys.exit(1)
    from config import WORK_DIR
    result = answer_question(sys.argv[1], WORK_DIR, sys.argv[2])
    print("\nAnswer:", result["answer"])
    print("\nRelevant moments:")
    for c in result["citations"]:
        print(f"  {format_time(c['start_ts'])} - {format_time(c['end_ts'])}  ({c['segment_id']})")