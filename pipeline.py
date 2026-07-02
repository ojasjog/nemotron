"""
End-to-end orchestrator for stage 1 (segmentation + captioning).

Usage:
    python pipeline.py /path/to/video.mp4 my_video_id

Produces:
    work/<video_id>/segments.json    -- segment boundaries + clip paths
    work/<video_id>/clips/*.mp4      -- extracted clips
    work/<video_id>/captions.jsonl   -- one structured caption per segment,
                                         ready to hand to the embedding/indexing stage
"""
import sys
import time

from config import WORK_DIR
from segment_video import segment_video
from captioner import caption_all


def run(video_path: str, video_id: str):
    t0 = time.time()
    print(f"[1/2] Segmenting {video_path} ...")
    segments = segment_video(video_path, video_id, WORK_DIR)
    print(f"      -> {len(segments)} segments in {time.time() - t0:.1f}s")

    t1 = time.time()
    print(f"[2/2] Captioning {len(segments)} segments ...")
    captions = caption_all(segments, WORK_DIR)
    print(f"      -> done in {time.time() - t1:.1f}s")

    print(f"\nTotal: {time.time() - t0:.1f}s. Output: work/{video_id}/captions.jsonl")
    return captions


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pipeline.py <video_path> <video_id>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])