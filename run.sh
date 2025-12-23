#!/bin/bash
# Airgarage Pipeline Runner - GPU Optimized

# Set CUDA paths
VENV_LIB="/workspace/venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="${VENV_LIB}/cudnn/lib:${VENV_LIB}/cublas/lib:${VENV_LIB}/cu13/lib:${LD_LIBRARY_PATH}"
export ORT_TENSORRT_UNAVAILABLE=1

# Activate venv
source /workspace/venv/bin/activate

# Run pipeline
python airgarage_advanced.py
