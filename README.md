### Run 

``vllm serve nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 \``
    ``--trust-remote-code --dtype bfloat16 \``
    ``--max-model-len 32768 --gpu-memory-utilization 0.90 \``
    ``--allowed-local-media-path /teamspace/studios/this_studio \``
    ``--served-model-name nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16``

### Followed by 

``python pipeline.py input_video.mp4 output_video_name``
