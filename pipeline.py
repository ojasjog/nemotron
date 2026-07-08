
from dotenv import load_dotenv
load_dotenv()
"""
End-to-end orchestrator: video in -> segments, captions, and a queryable
FAISS index out.

Usage:
    python pipeline.py /path/to/video.mp4 my_video_id

Produces:
    work/<video_id>/segments.json      -- segment boundaries + clip paths
    work/<video_id>/clips/*.mp4        -- extracted clips
    work/<video_id>/captions.jsonl     -- one structured caption per segment
    work/<video_id>/index.faiss        -- searchable embedding index
    work/<video_id>/index_meta.json    -- captions in index order (metadata store)

Once this finishes, ask questions with:
    python qa.py <video_id> "your question"
"""

import sys
import time

from config import WORK_DIR
from segment_video import segment_video
from captioner import caption_all
from embedder import build_index


def run(video_path: str, video_id: str):
    t0 = time.time()
    print(f"[1/3] Segmenting {video_path} ...")
    segments = segment_video(video_path, video_id, WORK_DIR)
    print(f"      -> {len(segments)} segments in {time.time() - t0:.1f}s")

    t1 = time.time()
    print(f"[2/3] Captioning {len(segments)} segments ...")
    captions = caption_all(segments, WORK_DIR)
    print(f"      -> done in {time.time() - t1:.1f}s")

    t2 = time.time()
    print(f"[3/3] Building search index ...")
    build_index(video_id, WORK_DIR)
    print(f"      -> done in {time.time() - t2:.1f}s")

    print(f"\nTotal: {time.time() - t0:.1f}s.")
    print(f'Ready. Ask questions with: python qa.py {video_id} "your question"')
    return captions


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pipeline.py <video_path> <video_id>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])