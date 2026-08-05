"""
Segments a video into short clips for captioning.

Same scene-detection + bounds logic as before. What changed:
- extract_clip() calls now run concurrently (ThreadPoolExecutor) instead of
  one ffmpeg subprocess at a time -- extraction is pure local CPU/disk work
  with zero dependency between clips, so this is a straightforward win.
- If config.PRIORITY_WINDOW is set, segments overlapping that window are
  submitted to the pool FIRST, so they tend to land on disk earliest.
"""
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

from config import (
    MIN_SEGMENT_SEC, MAX_SEGMENT_SEC, OVERLAP_SEC, SCENE_DETECT_THRESHOLD,
    EXTRACT_WORKERS, PRIORITY_WINDOW,
)


@dataclass
class Segment:
    segment_id: str
    video_id: str
    start_ts: float
    end_ts: float
    clip_path: str


def get_duration_sec(video_path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def detect_scene_cuts(video_path: str, threshold: float = SCENE_DETECT_THRESHOLD) -> list[float]:
    """Returns sorted list of timestamps (sec) where a scene change is detected."""
    scene_val = threshold / 100.0
    cmd = [
        "ffmpeg", "-i", video_path,
        "-filter:v", f"select='gt(scene,{scene_val})',showinfo",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    cuts = []
    for line in proc.stderr.splitlines():
        m = re.search(r"pts_time:(\d+\.?\d*)", line)
        if m:
            cuts.append(float(m.group(1)))
    return sorted(cuts)


def build_segment_bounds(duration: float, cuts: list[float]) -> list[tuple[float, float]]:
    """Turn raw cut points into segment (start, end) pairs honoring min/max length."""
    boundaries = [0.0] + cuts + [duration]
    boundaries = sorted(set(boundaries))

    merged = [boundaries[0]]
    for b in boundaries[1:]:
        if b - merged[-1] < MIN_SEGMENT_SEC:
            continue
        merged.append(b)
    if merged[-1] != duration:
        merged[-1] = duration

    final_bounds = []
    for start, end in zip(merged[:-1], merged[1:]):
        length = end - start
        if length <= MAX_SEGMENT_SEC:
            final_bounds.append((start, end))
        else:
            n_chunks = int(length // MAX_SEGMENT_SEC) + 1
            chunk_len = length / n_chunks
            for i in range(n_chunks):
                cs = start + i * chunk_len
                ce = min(start + (i + 1) * chunk_len, end)
                final_bounds.append((cs, ce))

    return final_bounds


def apply_overlap(bounds: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    out = []
    for i, (s, e) in enumerate(bounds):
        new_s = max(0.0, s - (OVERLAP_SEC if i > 0 else 0.0))
        new_e = min(duration, e + (OVERLAP_SEC if i < len(bounds) - 1 else 0.0))
        out.append((new_s, new_e))
    return out


def extract_clip(video_path: str, start: float, end: float, out_path: str):
    duration = end - start
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac",
        "-loglevel", "error",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def _overlaps_priority(start: float, end: float) -> bool:
    if PRIORITY_WINDOW is None:
        return False
    p_start, p_end = PRIORITY_WINDOW
    return start < p_end and end > p_start


def _ordered_indices(bounds: list[tuple[float, float]]) -> list[int]:
    """Priority-window segments first (in time order), then everything else
    (in time order). Submission order to the thread pool, not a guarantee of
    completion order, but with enough workers it strongly biases early."""
    priority = [i for i, (s, e) in enumerate(bounds) if _overlaps_priority(s, e)]
    rest = [i for i, (s, e) in enumerate(bounds) if not _overlaps_priority(s, e)]
    return priority + rest


def segment_video(video_path: str, video_id: str, work_dir: str) -> list[Segment]:
    clips_dir = Path(work_dir) / video_id / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    duration = get_duration_sec(video_path)
    cuts = detect_scene_cuts(video_path)
    bounds = build_segment_bounds(duration, cuts)
    bounds = apply_overlap(bounds, duration)

    # Pre-build Segment objects (order preserved = file order, needed later
    # for the manifest / index ordering), but extract concurrently.
    segments: list[Segment] = [None] * len(bounds)
    for i, (start, end) in enumerate(bounds):
        seg_id = f"{video_id}_seg{i:04d}"
        clip_path = str(clips_dir / f"{seg_id}.mp4")
        segments[i] = Segment(seg_id, video_id, round(start, 2), round(end, 2), clip_path)

    order = _ordered_indices(bounds)
    if PRIORITY_WINDOW is not None:
        n_priority = sum(1 for i in order if _overlaps_priority(*bounds[i]))
        print(f"      priority window {PRIORITY_WINDOW} -> {n_priority} segments extracted first")

    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as ex:
        futures = {
            ex.submit(extract_clip, video_path, bounds[i][0], bounds[i][1], segments[i].clip_path): i
            for i in order
        }
        done = 0
        for fut in as_completed(futures):
            fut.result()  # raises if ffmpeg failed for this clip
            done += 1
            if done % 10 == 0 or done == len(futures):
                print(f"      extracted {done}/{len(futures)} clips")

    manifest_path = Path(work_dir) / video_id / "segments.json"
    manifest_path.write_text(json.dumps([asdict(s) for s in segments], indent=2))
    return segments


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python segment_video.py <video_path> <video_id>")
        sys.exit(1)
    from config import WORK_DIR
    segs = segment_video(sys.argv[1], sys.argv[2], WORK_DIR)
    print(f"Created {len(segs)} segments:")
    for s in segs:
        print(f"  {s.segment_id}: {s.start_ts}s -> {s.end_ts}s  ({s.clip_path})")