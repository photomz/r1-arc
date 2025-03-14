uv run pyclean
uv run vllm serve unsloth/DeepSeek-R1-Distill-Qwen-14B 
# ssh -N -f -L 8000:localhost:8000 lambda