"""
Embeds each segment's caption into a vector and builds a FAISS index for the
video. This is what makes "when does X happen" style questions fast to
answer -- instead of re-reading every caption at query time, we do a
nearest-neighbor search over embeddings.

Each video gets its own isolated index (work/<video_id>/index.faiss +
work/<video_id>/index_meta.json), matching the "each upload is its own
index" requirement -- nothing here is shared across videos.
"""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME

_model = None  # lazy-loaded singleton so repeated calls in one process don't reload the model


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def caption_to_embedding_text(caption: dict) -> str:
    """Flattens a caption's structured fields into one string for embedding.
    Keeping this consistent between indexing and query time matters -- if you
    change this, rebuild existing indexes."""
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
    captions_path = Path(work_dir) / video_id / "captions.jsonl"
    captions = [json.loads(line) for line in captions_path.read_text().splitlines() if line.strip()]
    if not captions:
        raise ValueError(f"No captions found at {captions_path}")

    texts = [caption_to_embedding_text(c) for c in captions]
    embedder = get_embedder()
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on normalized vectors == cosine similarity
    index.add(vectors)

    index_dir = Path(work_dir) / video_id
    faiss.write_index(index, str(index_dir / "index.faiss"))
    # captions themselves double as the metadata store; order must match the
    # index exactly (position i in FAISS <-> captions[i])
    (index_dir / "index_meta.json").write_text(json.dumps(captions, indent=2))

    print(f"Indexed {len(captions)} segments for {video_id} -> {index_dir / 'index.faiss'}")
    return index, captions


def load_index(video_id: str, work_dir: str):
    index_dir = Path(work_dir) / video_id
    index = faiss.read_index(str(index_dir / "index.faiss"))
    captions = json.loads((index_dir / "index_meta.json").read_text())
    return index, captions


def search(video_id: str, work_dir: str, query: str, k: int) -> list[dict]:
    """Returns the top-k caption dicts most semantically similar to the query,
    each annotated with a similarity score."""
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