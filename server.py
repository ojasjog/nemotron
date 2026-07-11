"""
server.py -- thin FastAPI wrapper around the nemotron video pipeline.

Drop this file (and the static/ folder next to it) into the ROOT of the
ojasjog/nemotron checkout, i.e. alongside pipeline.py, qa.py, config.py,
segment_video.py, captioner.py, embedder.py.

Workflow exposed to the browser:
  1. POST /api/upload      -- upload a video file, kicks off segmentation +
                               captioning + indexing in a background thread
  2. GET  /api/status/{id} -- poll processing status
  3. GET  /api/video/{id}  -- stream the uploaded video back (for playback +
                               seeking to cited timestamps)
  4. POST /api/ask         -- ask a question about a processed video

Run with:
    pip install fastapi "uvicorn[standard]" python-multipart
    uvicorn server:app --reload --port 8080

Then open http://localhost:8080
"""

from __future__ import annotations

import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import WORK_DIR
from segment_video import segment_video
from captioner import caption_all
from embedder import build_index
from qa import answer_question

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
Path(WORK_DIR).mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}

app = FastAPI(title="nemotron video QA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# video_id -> job state
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _set_job(video_id: str, **kwargs: Any) -> None:
    with JOBS_LOCK:
        JOBS.setdefault(video_id, {}).update(kwargs)


def _get_job(video_id: str) -> dict[str, Any] | None:
    with JOBS_LOCK:
        job = JOBS.get(video_id)
        return dict(job) if job else None


# --------------------------------------------------------------------------
# Background processing
# --------------------------------------------------------------------------

def _run_pipeline(video_path: str, video_id: str) -> None:
    t0 = time.time()
    try:
        _set_job(video_id, status="segmenting", message="Detecting scene cuts and slicing clips...")
        segments = segment_video(video_path, video_id, WORK_DIR)
        _set_job(video_id, num_segments=len(segments))

        if not segments:
            _set_job(video_id, status="error", message="No segments were produced from this video.")
            return

        _set_job(
            video_id,
            status="captioning",
            message=f"Captioning {len(segments)} clips with Nemotron VL...",
        )
        caption_all(segments, WORK_DIR)

        _set_job(video_id, status="indexing", message="Building the search index...")
        build_index(video_id, WORK_DIR)

        _set_job(
            video_id,
            status="ready",
            message=f"Ready. Processed {len(segments)} clips in {time.time() - t0:.0f}s.",
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _set_job(video_id, status="error", message=f"{type(exc).__name__}: {exc}")


def _slugify(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return stem or "video"


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

class AskRequest(BaseModel):
    video_id: str
    question: str


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use one of {sorted(ALLOWED_EXTENSIONS)}.")

    video_id = f"{_slugify(file.filename)}-{uuid.uuid4().hex[:8]}"
    dest = UPLOAD_DIR / f"{video_id}{ext}"

    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    _set_job(
        video_id,
        status="queued",
        message="Upload received, starting processing...",
        filename=file.filename,
        video_path=str(dest),
        num_segments=0,
    )

    thread = threading.Thread(target=_run_pipeline, args=(str(dest), video_id), daemon=True)
    thread.start()

    return {"video_id": video_id, "filename": file.filename}


@app.get("/api/status/{video_id}")
async def get_status(video_id: str) -> dict[str, Any]:
    job = _get_job(video_id)
    if job is None:
        raise HTTPException(404, "Unknown video_id")
    return {
        "video_id": video_id,
        "status": job.get("status"),
        "message": job.get("message"),
        "num_segments": job.get("num_segments", 0),
        "filename": job.get("filename"),
    }


@app.get("/api/video/{video_id}")
async def get_video(video_id: str) -> FileResponse:
    job = _get_job(video_id)
    if job is None or not job.get("video_path"):
        raise HTTPException(404, "Unknown video_id")
    return FileResponse(job["video_path"])


@app.post("/api/ask")
async def ask(req: AskRequest) -> dict[str, Any]:
    job = _get_job(req.video_id)
    if job is None:
        raise HTTPException(404, "Unknown video_id")
    if job.get("status") != "ready":
        raise HTTPException(409, f"Video isn't ready yet (status: {job.get('status')}).")
    if not req.question.strip():
        raise HTTPException(400, "Question can't be empty.")

    result = answer_question(req.video_id, WORK_DIR, req.question.strip())
    return result


# --------------------------------------------------------------------------
# Static frontend
# --------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=str(APP_DIR / "static"), html=True), name="static")