uv run pyclean src
uv run vllm serve unsloth/DeepSeek-R1-Distill-Qwen-1.5B \
  --quantization bitsandbytes \
  --load-format bitsandbytes
# ssh -N -f -L 8000:localhost:8000 lambda