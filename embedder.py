"""
Embeds each segment's caption into a vector and builds a FAISS index for the
video.

What's new: an incremental path (IncrementalIndex) that lets pipeline.py add
one caption at a time as captioning completes, and flush to disk
periodically -- so a second process can run qa.py against a partial index
while the rest of the file is still being captioned. build_index() (the
original one-shot batch version) is kept as-is for backward compatibility /
standalone use (`python embedder.py <video_id>`).
"""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME

_model = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def caption_to_embedding_text(caption: dict) -> str:
    parts = [caption.get("description", "")]
    if caption.get("objects"):
        parts.append("Objects: " + ", ".join(caption["objects"]))
    if caption.get("actions"):
        parts.append("Actions: " + ", ".join(caption["actions"]))
    if caption.get("on_screen_text"):
        parts.append("Text visible: " + caption["on_screen_text"])
    if caption.get("setting"):
        parts.append("Setting: " + caption["setting"])
    return ". ".join(p for p in parts if p)


def build_index(video_id: str, work_dir: str):
    """Original one-shot batch build -- reads all captions, embeds, indexes."""
    captions_path = Path(work_dir) / video_id / "captions.jsonl"
    captions = [json.loads(line) for line in captions_path.read_text().splitlines() if line.strip()]
    if not captions:
        raise ValueError(f"No captions found at {captions_path}")

    texts = [caption_to_embedding_text(c) for c in captions]
    embedder = get_embedder()
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_dir = Path(work_dir) / video_id
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "index_meta.json").write_text(json.dumps(captions, indent=2))

    print(f"Indexed {len(captions)} segments for {video_id} -> {index_dir / 'index.faiss'}")
    return index, captions


class IncrementalIndex:
    """Add captions one at a time; flush() writes a valid, loadable
    index.faiss + index_meta.json at any point -- qa.py doesn't need to know
    or care whether the index is "done"."""

    def __init__(self, video_id: str, work_dir: str):
        self.video_id = video_id
        self.work_dir = work_dir
        self.embedder = get_embedder()
        self.index = None  # created lazily once we know the embedding dim
        self.captions: list[dict] = []

    def add(self, caption: dict):
        text = caption_to_embedding_text(caption)
        vec = self.embedder.encode([text], normalize_embeddings=True)
        vec = np.asarray(vec, dtype="float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(vec.shape[1])
        self.index.add(vec)
        self.captions.append(caption)

    def flush(self):
        if self.index is None:
            return  # nothing added yet
        index_dir = Path(self.work_dir) / self.video_id
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "index.faiss"))
        (index_dir / "index_meta.json").write_text(json.dumps(self.captions, indent=2))


def load_index(video_id: str, work_dir: str):
    index_dir = Path(work_dir) / video_id
    index = faiss.read_index(str(index_dir / "index.faiss"))
    captions = json.loads((index_dir / "index_meta.json").read_text())
    return index, captions


def search(video_id: str, work_dir: str, query: str, k: int) -> list[dict]:
    index, captions = load_index(video_id, work_dir)
    embedder = get_embedder()
    q_vec = embedder.encode([query], normalize_embeddings=True)
    q_vec = np.asarray(q_vec, dtype="float32")

    scores, indices = index.search(q_vec, min(k, len(captions)))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        cap = dict(captions[idx])
        cap["score"] = float(score)
        results.append(cap)
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python embedder.py <video_id>  (run pipeline.py first)")
        sys.exit(1)
    from config import WORK_DIR
    build_index(sys.argv[1], WORK_DIR)