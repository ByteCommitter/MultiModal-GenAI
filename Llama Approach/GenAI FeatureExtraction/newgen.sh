#!/bin/bash
#SBATCH --job-name=generative-multimodal
#SBATCH --output=logs/generative_job_%j.out
#SBATCH --partition=gpu_a100_8          # Changed to A100 which has more availability
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00                 # Reduced from 24h (start conservative)
#SBATCH --mem=48G                       # Reduced from 96G (sufficient for this task)
#SBATCH --cpus-per-task=6               # Reduced from 12 (main issue causing pending)

### --- 1. Load available modules ---
module purge

# Load CUDA module - attempt 12.1, if unavailable load latest
if module spider cuda/12.1 >/dev/null 2>&1; then
    module load cuda/12.1
else
    latest_cuda=$(module spider cuda | grep -Eo "cuda/[0-9]+\.[0-9]+" | sort -V | tail -n 1)
    module load $latest_cuda
fi

# Load GCC module - attempt 11.2.0, if unavailable load latest
if module spider gcc/11.2.0 >/dev/null 2>&1; then
    module load gcc/11.2.0
else
    latest_gcc=$(module spider gcc | grep -Eo "gcc/[0-9]+\.[0-9]+" | sort -V | tail -n 1)
    module load $latest_gcc
fi

# Load Python module - attempt 3.10, if unavailable load latest
if module spider python/3.10 >/dev/null 2>&1; then
    module load python/3.10
else
    latest_python=$(module spider python | grep -Eo "python/[0-9]+\.[0-9]+" | sort -V | tail -n 1)
    module load $latest_python
fi

### --- 2. Setup CUDA environment ---
if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export PATH=$CUDA_HOME/bin:$PATH
fi

# Environment variables for stable training
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

### --- 3. Print system information ---
echo "=== Generative Multi-Modal Training System Info ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
echo "CUDA Version: $(nvcc --version | grep release)"
echo "Python Version: $(python --version)"
echo "===================================================="

### --- 4. Activate your virtual environment ---
source ~/.bashrc
conda activate llama-env

### --- 5. Verify dependencies ---
echo "Verifying dependencies..."
pip install -q --upgrade torch torchvision torchaudio
pip install -q transformers>=4.30.0
pip install -q scikit-learn pandas numpy tqdm
echo "Dependencies verified!"

### --- 6. Create logs directory ---
mkdir -p logs
mkdir -p trained_models

### --- 7. Run the generative training script ---
echo ""
echo "=========================================================="
echo "Starting Generative Multi-Modal Training..."
echo "=========================================================="
echo ""

# Run the improved generative model script
srun python newgen.py 2>&1 | tee logs/generative_training_$(date +%Y%m%d_%H%M%S).log

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=========================================================="
    echo "✅ Training completed successfully!"
    echo "=========================================================="
    echo "Completion time: $(date)"
    echo "Check trained_models/ for saved models"
    echo "Check logs/ for detailed training logs"
    echo "=========================================================="
else
    echo ""
    echo "=========================================================="
    echo "❌ Training failed with exit code ${PIPESTATUS[0]}"
    echo "=========================================================="
    echo "Check the log file for error details"
    exit 1
fi
