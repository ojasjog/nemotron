"""
Segments a video into short clips for captioning.

Approach:
1. Use ffmpeg's built-in scene-detection filter to find cut points (no extra
   heavyweight deps like PySceneDetect needed -- ffmpeg is already required
   for clip extraction, so we reuse it).
2. Merge/split cuts so every segment falls within [MIN_SEGMENT_SEC, MAX_SEGMENT_SEC].
3. Add a small overlap between consecutive segments so actions spanning a
   cut boundary aren't lost.
4. Extract each segment as its own mp4 clip (re-encoded, not stream-copied,
   so cut points land exactly where we ask -- stream copy snaps to keyframes
   and can be off by seconds).

Output: list[Segment] with start/end timestamps + path to the clip file.
"""
import json
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

from config import MIN_SEGMENT_SEC, MAX_SEGMENT_SEC, OVERLAP_SEC, SCENE_DETECT_THRESHOLD


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
    # ffmpeg scene filter reports a score 0-100 per frame; we threshold it via
    # select and read the pts_time back out of showinfo's stderr log.
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

    # First pass: merge boundaries that create too-short segments
    merged = [boundaries[0]]
    for b in boundaries[1:]:
        if b - merged[-1] < MIN_SEGMENT_SEC:
            continue  # skip this boundary, extends current segment
        merged.append(b)
    if merged[-1] != duration:
        merged[-1] = duration  # ensure we end exactly at video end

    # Second pass: split any segment that's too long
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
        "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
        "-loglevel", "error",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def segment_video(video_path: str, video_id: str, work_dir: str) -> list[Segment]:
    clips_dir = Path(work_dir) / video_id / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    duration = get_duration_sec(video_path)
    cuts = detect_scene_cuts(video_path)
    bounds = build_segment_bounds(duration, cuts)
    bounds = apply_overlap(bounds, duration)

    segments = []
    for i, (start, end) in enumerate(bounds):
        seg_id = f"{video_id}_seg{i:04d}"
        clip_path = str(clips_dir / f"{seg_id}.mp4")
        extract_clip(video_path, start, end, clip_path)
        segments.append(Segment(seg_id, video_id, round(start, 2), round(end, 2), clip_path))

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