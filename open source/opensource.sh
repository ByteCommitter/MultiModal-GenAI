#!/bin/bash
#SBATCH --job-name=emotion-detection    # Name your job
#SBATCH --output=logs/emotion_%j.out    # Log output file
#SBATCH --partition=gpu_a100_8          # SLURM partition to run on
#SBATCH --nodelist=gpunode4             # Specific node (optional)
#SBATCH --gres=gpu:1                    # Number of GPUs
#SBATCH --time=8:00:00                  # Max wall time
#SBATCH --mem=32G                       # Memory
#SBATCH --cpus-per-task=4              # CPU cores

### --- 1. Load modules ---

module purge

# Load CUDA module
if module spider cuda/12.1 >/dev/null 2>&1; then
    module load cuda/12.1
else
    latest_cuda=$(module spider cuda | grep -Eo "cuda/[0-9]+\.[0-9]+" | sort -V | tail -n 1)
    module load $latest_cuda
fi

# Load Python module
if module spider python/3.10 >/dev/null 2>&1; then
    module load python/3.10
else
    latest_python=$(module spider python | grep -Eo "python/[0-9]+\.[0-9]+" | sort -V | tail -n 1)
    module load $latest_python
fi

### --- 2. Setup environment ---

export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false

### --- 3. Print system information ---
echo "=== Emotion Detection Training System Info ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
echo "Python Version: $(python --version)"
echo "=================================================="

### --- 4. Activate virtual environment ---
source ~/.bashrc
conda activate llama-env

### --- 5. Install dependencies ---
echo "Installing/verifying dependencies..."

pip install -q torch torchvision torchaudio
pip install -q transformers>=4.30.0
pip install -q scikit-learn pandas numpy tqdm
pip install -q wandb

echo "Dependencies installed successfully!"

### --- 6. Create directories ---
mkdir -p logs
mkdir -p trained_models

### --- 7. Run training script ---
echo "Starting Emotion Detection Training..."
echo "======================================="

export PYTHONUNBUFFERED=1

srun python opensource.py 2>&1 | tee logs/opensource_$(date +%Y%m%d_%H%M%S).log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "======================================="
    echo "✅ Training completed successfully!"
    echo "💾 Check trained_models/ for saved models"
    echo "📋 Check logs/ for training logs"
else
    echo "❌ Training failed with exit code ${PIPESTATUS[0]}"
    echo "📋 Check logs for error details"
    exit 1
fi

echo "======================================="
echo "🎉 Job completed at: $(date)"
