#!/bin/bash

# Clean up any hung processes
pkill -f airgarage_advanced.py 2>/dev/null

# Clear GPU memory (optional)
nvidia-smi --gpu-reset 2>/dev/null || true

# Wait a moment
sleep 2

# Activate venv and run
echo "Starting pipeline with optimized settings (CPU mode)..."
source /workspace/venv/bin/activate
python /workspace/airgarage_advanced.py
