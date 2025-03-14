#!/bin/bash

counter=1

while true; do
    echo "==============================================="
    echo "[$(date)] Starting training script (Run #$counter)"
    echo "==============================================="

    uv run pyclean src   
    uv run python -m src.playpen.trainer_qwen_7b
    
    EXIT_CODE=$?
    echo "[$(date)] Run #$counter exited with $EXIT_CODE. Restarting..."

    # Kill CUDA memory leaks (assume we're only tenant of this GPU)
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read -r pid; do
        if [ ! -z "$pid" ]; then
            kill "$pid"
            echo "Killed GPU process $pid"
        fi
    done

    ((counter++))
    
    sleep 5
done