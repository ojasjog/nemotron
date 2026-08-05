from dotenv import load_dotenv
load_dotenv()
"""
End-to-end orchestrator: video in -> segments, captions, and a queryable
FAISS index out.

Usage:
    python pipeline.py /path/to/video.mp4 my_video_id

What changed from the original:
- Extraction and captioning both run concurrently now (see segment_video.py
  / captioner.py) instead of one clip / one API call at a time.
- The FAISS index is built incrementally as captions come back, and flushed
  to disk every config.INDEX_FLUSH_EVERY captions -- so you can open a
  second terminal and run `python qa.py <video_id> "..."` while this is
  still running, instead of waiting for the whole file to finish.
- If config.PRIORITY_WINDOW is set, segments overlapping that window are
  extracted and captioned first, and you'll see a message the moment that
  window is fully indexed -- that's your "safe to start investigating" point,
  well before the rest of the file is done.

Produces (same layout as before):
    work/<video_id>/segments.json
    work/<video_id>/clips/*.mp4
    work/<video_id>/captions.jsonl
    work/<video_id>/index.faiss
    work/<video_id>/index_meta.json
"""

import sys
import time
from pathlib import Path

from config import WORK_DIR, PRIORITY_WINDOW, INDEX_FLUSH_EVERY
from segment_video import segment_video
from captioner import caption_all
from embedder import IncrementalIndex


def run(video_path: str, video_id: str):
    t0 = time.time()

    print(f"[1/3] Segmenting {video_path} ...")
    segments = segment_video(video_path, video_id, WORK_DIR)
    print(f"      -> {len(segments)} segments in {time.time() - t0:.1f}s")

    # truncate captions.jsonl since caption_all() now appends per-completion
    (Path(WORK_DIR) / video_id / "captions.jsonl").write_text("")

    idx = IncrementalIndex(video_id, WORK_DIR)
    priority_ready_announced = PRIORITY_WINDOW is None  # nothing to announce if unset
    priority_seg_ids = set()
    if PRIORITY_WINDOW is not None:
        p_start, p_end = PRIORITY_WINDOW
        priority_seg_ids = {
            s.segment_id for s in segments if s.start_ts < p_end and s.end_ts > p_start
        }

    since_flush = 0

    def on_caption(cap, _pos_in_file_order):
        nonlocal since_flush, priority_ready_announced
        idx.add(cap.__dict__)
        since_flush += 1
        if since_flush >= INDEX_FLUSH_EVERY:
            idx.flush()
            since_flush = 0

        if not priority_ready_announced:
            priority_seg_ids.discard(cap.segment_id)
            if not priority_seg_ids:
                idx.flush()
                since_flush = 0
                elapsed = time.time() - t0
                print(
                    f"\n      >>> priority window fully indexed at {elapsed:.1f}s -- "
                    f"you can run `python qa.py {video_id} \"...\"` now. "
                    f"Remaining segments continue in the background.\n"
                )
                priority_ready_announced = True

    t1 = time.time()
    print(f"[2/3] Captioning {len(segments)} segments ...")
    caption_all(segments, WORK_DIR, on_caption=on_caption)
    print(f"      -> done in {time.time() - t1:.1f}s")

    t2 = time.time()
    idx.flush()  # final flush, guaranteed complete regardless of INDEX_FLUSH_EVERY timing
    print(f"[3/3] Final index flush -> {time.time() - t2:.1f}s")

    print(f"\nTotal: {time.time() - t0:.1f}s.")
    print(f'Ready. Ask questions with: python qa.py {video_id} "your question"')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pipeline.py <video_path> <video_id>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])