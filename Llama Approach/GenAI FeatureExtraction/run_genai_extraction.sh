#!/bin/bash
#SBATCH --job-name=genai_daic
#SBATCH --output=/home/dipanjan/rugraj/DIAC-WOZ/logs/genai_extraction_%j.out
#SBATCH --error=/home/dipanjan/rugraj/DIAC-WOZ/logs/genai_extraction_%j.err
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=f20220226@hyderabad.bits-pilani.ac.in

# ============================================================================
# SLURM Job Script: GenAI Feature Extraction for DAIC-WOZ
# Partition: gpu_h100_4 (H100 GPUs) - fallback to gpu_a100_8 if needed
# ============================================================================

echo "========================================================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================================================"

# Create logs directory if it doesn't exist
mkdir -p /home/dipanjan/rugraj/DIAC-WOZ/logs

# Set working directory
cd /home/dipanjan/rugraj/DIAC-WOZ/ || exit 1

# Environment setup
echo ""
echo "Setting up environment..."
echo "----------------------------------------"

# Set HuggingFace cache
export HF_HOME=/home/dipanjan/.cache/huggingface
export TRANSFORMERS_CACHE=/home/dipanjan/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0

# Activate conda environment
echo "Activating conda environment..."
source /opt/conda/etc/profile.d/conda.sh  # Adjust path if needed
conda activate base

# Verify environment
echo ""
echo "Environment check:"
echo "----------------------------------------"
python --version
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo "GPU memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

# Check required files
echo ""
echo "Checking required files..."
echo "----------------------------------------"
ls -lh daic_metadata.csv
ls -lh acoustic_features.csv
ls -lh visual_features.csv
ls -lh genai_extraction.py
ls -lh genai_extraction_llama3.py
ls -lh genai_multimodal.py

# Run the extraction script
echo ""
echo "========================================================================"
echo "Starting GenAI feature extraction..."
echo "========================================================================"
echo ""

# Run with output logging
python genai_hierarchical_extraction.py 2>&1 | tee genai_extraction_runtime.log

EXIT_CODE=$?

# Summary
echo ""
echo "========================================================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================================================"

# Check outputs
if [ -f "genai_features.csv" ]; then
    echo ""
    echo "✓ Output generated successfully:"
    ls -lh genai_features.csv
    echo ""
    echo "Preview:"
    head -5 genai_features.csv
else
    echo ""
    echo "✗ ERROR: Output file not generated!"
    echo "Check error log: logs/genai_extraction_${SLURM_JOB_ID}.err"
fi

# Display error log if exists
if [ -f "genai_extraction_errors.log" ]; then
    ERROR_COUNT=$(wc -l < genai_extraction_errors.log)
    echo ""
    echo "⚠️  Errors encountered: $ERROR_COUNT"
    echo "First 10 errors:"
    head -10 genai_extraction_errors.log
fi

# GPU stats
echo ""
echo "Final GPU status:"
nvidia-smi

exit $EXIT_CODE
