"""
Central configuration for the video segmentation + captioning stage.
Tune these once you see how your videos behave.
"""

# --- Segmentation ---
MIN_SEGMENT_SEC = 6          # merge scenes shorter than this into neighbors
MAX_SEGMENT_SEC = 25         # split scenes longer than this (Nemotron VL video input is capped ~2min anyway)
OVERLAP_SEC = 2.0            # small overlap so an action spanning a cut isn't lost
SCENE_DETECT_THRESHOLD = 27.0  # PySceneDetect ContentDetector threshold; lower = more sensitive to cuts

# --- Captioning / Nemotron VL server (vLLM, OpenAI-compatible) ---
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"          # vLLM ignores this, OpenAI client just requires a non-empty string
MODEL_NAME = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"  # match --served-model-name on your vllm serve command

CAPTION_TEMPERATURE = 0.0
CAPTION_MAX_TOKENS = 512
CAPTION_SYSTEM_PROMPT = "/no_think"  # video inputs don't support reasoning mode on this model; keep it off
CAPTION_RETRIES = 2

# --- Paths ---
WORK_DIR = "work"            # where per-video segment clips + caption jsonl are written