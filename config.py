"""
Central configuration for the video segmentation + captioning stage.
Tune these once you see how your videos behave.
"""
"""
Append these to your existing config.py. Nothing above is changed --
these are new knobs only used by the concurrent pipeline additions.
"""

# --- Concurrency ---
# Clip extraction is local ffmpeg work -- bound by CPU cores, not network.
# Rule of thumb: (cpu_count - 1), capped somewhere reasonable.
EXTRACT_WORKERS = 4

# Captioning is bound by the hosted API's rate limit, not your CPU.
# Start conservative and raise it until you see 429s, then back off.
CAPTION_WORKERS = 8

# --- Priority window ---
# If set, segments overlapping this (start_sec, end_sec) window are
# extracted + captioned + indexed FIRST, so you can start querying the
# actually-relevant footage before the rest of the file finishes.
# Leave as None to process in file order (oldest behavior).
PRIORITY_WINDOW = None  # e.g. (2400.0, 4200.0) for a window from 40:00-70:00

# --- Incremental indexing ---
# Flush the FAISS index to disk every N newly-captioned segments, so a
# second process (e.g. `python qa.py <video_id> "..."`) can query against
# partial results while captioning continues in the background.
INDEX_FLUSH_EVERY = 5

# --- Segmentation ---
MIN_SEGMENT_SEC = 6          # merge scenes shorter than this into neighbors
MAX_SEGMENT_SEC = 25         # split scenes longer than this (Nemotron VL video input is capped ~2min anyway)
OVERLAP_SEC = 2.0            # small overlap so an action spanning a cut isn't lost
SCENE_DETECT_THRESHOLD = 27.0  # PySceneDetect ContentDetector threshold; lower = more sensitive to cuts

# --- Captioning backend ---
# "local"  -> your own vLLM server (e.g. on an L40S/A100/H100)
# "hosted" -> NVIDIA's hosted API, offloads all GPU work off your machine.
#             Needed on GPUs Nemotron VL isn't built for (e.g. Colab T4,
#             which is Turing architecture -- no FP8 tensor cores, and only
#             16GB VRAM vs. the ~24GB+ the 12B model needs in BF16).
CAPTION_BACKEND = "hosted"  # "local" or "hosted"

# --- Local vLLM server (only used if CAPTION_BACKEND == "local") ---
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"          # vLLM ignores this, OpenAI client just requires a non-empty string
MODEL_NAME = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
# match --served-model-name on your vllm serve command

# --- Hosted NVIDIA API (only used if CAPTION_BACKEND == "hosted") ---
# Get a free trial key at https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl
# then: import os; os.environ["NVIDIA_API_KEY"] = "nvapi-..."  (in Colab, use a secret/cell, don't hardcode it)
HOSTED_BASE_URL = "https://integrate.api.nvidia.com/v1"
HOSTED_MODEL_NAME = "nvidia/nemotron-nano-12b-v2-vl"
HOSTED_API_KEY_ENV_VAR = "NVIDIA_API_KEY"

QA_BACKEND = "hosted"  # or "local"
QA_MODEL_NAME = "meta/llama-3.1-8b-instruct"       # local vLLM served name
QA_HOSTED_MODEL_NAME = "meta/llama-3.1-8b-instruct"  # e.g. via NVIDIA build API, Groq, Together

# --- Embedding / FAISS retrieval ---
# Small, fast, well-supported sentence embedding model. Swap for
# "BAAI/bge-small-en-v1.5" if you want a quality bump at a bit more cost.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5              # segments retrieved for a normal question
TOP_K_WIDE = 10         # segments retrieved for existence-style questions ("is there a forklift")

# --- Answer synthesis (text-only LLM call, no video) ---
# Reuses the same backend/model as captioning (config.CAPTION_BACKEND) --
# text-only calls are cheap, no need for a separate model.
QA_TEMPERATURE = 0.0
QA_MAX_TOKENS = 400



CAPTION_TEMPERATURE = 0.0
CAPTION_MAX_TOKENS = 512
CAPTION_SYSTEM_PROMPT = "/no_think"  # video inputs don't support reasoning mode on this model; keep it off
CAPTION_RETRIES = 2

# How densely to sample frames from each (already-short) segment clip.
# Higher fps = better chance of catching brief actions/on-screen text, at the
# cost of more tokens/slower inference. 4 fps on a 6-25s clip is 24-100 frames,
# which is well within budget for short clips. Bump toward 8 if you're still
# missing fast events; drop toward 1-2 if latency becomes a problem.
CAPTION_FPS = 4.0

# --- Paths ---
WORK_DIR = "work"            # where per-video segment clips + caption jsonl are written