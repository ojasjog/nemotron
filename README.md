# Stage 1: Segmentation + Captioning

Turns one video into a set of short clips, each captioned as structured JSON
(description, people_count, objects, actions, on-screen text, setting) with
timestamps. This JSONL is the input to the embedding/FAISS stage next.

## Running on Colab (T4)

T4 is not on Nemotron VL's supported hardware list (that's L40S/A100/H100/B200-class GPUs -- Ampere or newer). A T4 lacks the tensor core support the FP8/NVFP4 checkpoints need and doesn't have enough VRAM for BF16 weights, so self-hosting via vLLM on a Colab T4 will not work reliably.

Instead, set `CAPTION_BACKEND = "hosted"` in `config.py` (this is now the default) and use NVIDIA's hosted API:

```python
# In a Colab cell, before running the pipeline:
import os
os.environ["NVIDIA_API_KEY"] = "nvapi-..."  # get a free key at
# https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl
```

Then just run `python pipeline.py your_video.mp4 video_001` as normal -- the T4 only needs to do ffmpeg segmentation and HTTP calls, both trivial for it. All the actual VLM inference happens on NVIDIA's servers.

If you hit a payload-too-large error from the hosted API (video clips are sent as base64), shrink `MAX_SEGMENT_SEC` in `config.py` or lower the encode resolution/bitrate in `segment_video.py`'s `extract_clip()`.

## Running with a local vLLM server (L40S/A100/H100/etc.)

Set `CAPTION_BACKEND = "local"` in `config.py`, then:

```bash
pip install "vllm" --extra-index-url https://wheels.vllm.ai/nightly --prerelease=allow
# or: docker pull vllm/vllm-openai:nightly-8bff831f0aa239006f34b721e63e1340e3472067

vllm serve nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 \
    --trust-remote-code --dtype bfloat16 \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --allowed-local-media-path /home/claude/video_qa_pipeline \
    --served-model-name nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 \
    --video-pruning-rate 0
```

`--allowed-local-media-path` must cover the directory this pipeline writes
clips into (`work/<video_id>/clips/`), or vLLM will refuse to read the
`file://` URLs the captioner sends it. If you're tight on VRAM, switch to the
FP8 checkpoint (`nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8`) instead of BF16.

## Install pipeline deps

```bash
pip install -r requirements.txt
```

(ffmpeg/ffprobe must be on PATH -- already present in most Lightning AI
Studio images.)

## 3. Run

```bash
python pipeline.py /path/to/uploaded_video.mp4 video_001
```

Output:
```
work/video_001/segments.json     # segment boundaries + clip paths
work/video_001/clips/*.mp4       # extracted clips
work/video_001/captions.jsonl    # one structured caption per segment
```

Each line of `captions.jsonl` looks like:

```json
{"segment_id": "video_001_seg0004", "video_id": "video_001", "start_ts": 42.1, "end_ts": 61.3,
 "description": "A forklift moves a pallet across the warehouse floor while a worker in a hi-vis vest watches.",
 "people_count": 1, "objects": ["forklift", "pallet", "hi-vis vest"],
 "actions": ["forklift moves pallet", "worker observes"],
 "on_screen_text": "", "setting": "warehouse interior"}
```

## Tuning knobs (config.py)

- `MIN_SEGMENT_SEC` / `MAX_SEGMENT_SEC` -- segment length bounds. Shorter =
  more precise timestamps but weaker per-segment context; longer = richer
  captions but blurrier "when exactly" answers. 6-25s is a reasonable start.
- `OVERLAP_SEC` -- overlap between consecutive segments so an action that
  straddles a cut isn't missed by either segment.
- `SCENE_DETECT_THRESHOLD` -- ffmpeg scene-cut sensitivity (0-100). Lower
  catches more/smaller cuts; raise it if you're getting too many segments on
  a mostly-static video.

## Improving captioning accuracy

If captions feel imprecise (missed events, wrong counts, blurry timing), work through these roughly in order:

1. **Disable EVS frame pruning on the server.** By default vLLM's Efficient
   Video Sampling drops frames it judges redundant to save compute -- exactly
   the frames that might contain the brief action you care about. Add
   `--video-pruning-rate 0` to your `vllm serve` command.
2. **Increase frame sampling density per request.** `config.CAPTION_FPS`
   (default 4.0) controls how many frames per second are pulled from each
   clip via `media_io_kwargs`. Since segments are already short, bumping this
   toward 8 is cheap and can catch fast/brief events. Drop it back down if
   latency becomes a problem.
3. **Shorten `MAX_SEGMENT_SEC` / tighten `SCENE_DETECT_THRESHOLD`.** Smaller
   segments mean less for the model to compress into one caption, and
   timestamps become more precise by construction.
4. **Cross-check counts/objects with a second pass or a dedicated model.**
   VLM captions are the weakest link for exact counting ("how many people")
   and small/occluded object detection. If that precision matters a lot,
   consider a lightweight object detector (e.g. YOLO) run on keyframes
   specifically for counting, with the VLM caption used for everything else.
5. **Re-run with a higher CRF if fine text is getting missed** (already set
   to `-crf 18`, up from ffmpeg's default 23, in `segment_video.py`) --
   compression artifacts can blur small signage text enough to break
   OCR-style reading.

## Asking questions

Once `pipeline.py` finishes (segments + captions + index all built):

```bash
python qa.py video_001 "Is there a forklift?"
python qa.py video_001 "When does someone enter the frame?"
python qa.py video_001 "How many people appear in total?"
python qa.py video_001 "What does the sign say?"
```

Each call returns an `answer` string plus a `citations` list of
`{segment_id, start_ts, end_ts}` -- that's what a frontend uses to render
"jump to this moment" links/timestamps next to the answer.

**How it decides what to do with a question:**
- "How many X in total" style questions bypass vector search entirely and
  scan every segment's structured `people_count` field, since aggregation
  across the whole video isn't something similarity search does well. The
  answer explicitly caveats that this is a max-per-segment estimate, not a
  deduplicated total (see "Known gaps" above -- no cross-segment identity
  tracking yet).
- "Is there / are there" existence questions retrieve a wider set of
  candidate segments (`TOP_K_WIDE`, default 10) before asking the LLM,
  since ruling something in/out needs to look at more of the video than a
  single best-matching moment.
- Everything else does a normal top-k semantic search (`TOP_K`, default 5)
  over the FAISS index, then hands the retrieved captions + timestamps to
  the LLM to write a direct answer with citations.

## Known gaps to address in later stages

- **People counting across the whole video**: each segment reports its own
  `people_count`, but there's no cross-segment identity tracking here, so
  "how many people appear in total" needs aggregation logic (or a dedicated
  tracking pass) in the retrieval/answer stage -- summing per-segment counts
  will overcount someone who appears in multiple segments.
- **On-screen text**: the VL model's OCR is decent but not specialized. If
  sign/label reading needs to be highly reliable, consider adding a
  dedicated OCR pass (e.g. PaddleOCR on keyframes) alongside `on_screen_text`.
- **Failed captions**: if JSON parsing fails after retries, the segment is
  still recorded with a `[CAPTION PARSE FAILED...]` description rather than
  being silently dropped -- worth surfacing these in a QA pass before indexing.