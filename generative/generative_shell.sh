#!/bin/bash
#SBATCH --job-name=gen-multimodal            # Name your job
#SBATCH --output=logs/gen_job_%j.out         # Log output file
#SBATCH --partition=gpu_a100_8               # SLURM partition to run on
#SBATCH --nodelist=gpunode4                  # Specific node (optional)
#SBATCH --gres=gpu:1                         # Number of GPUs
#SBATCH --time=10:00:00                      # Max wall time
#SBATCH --mem=48G                            # Memory
#SBATCH --cpus-per-task=6                    # CPU cores

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

export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDA_ARCH_LIST="8.0"
export TOKENIZERS_PARALLELISM=false

### --- 3. Print system information ---
echo "=== Generative Multi-Modal Training System Info ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
echo "CUDA Version: $(nvcc --version | grep release)"
echo "Python Version: $(python --version)"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
echo "=================================================="

### --- 4. Activate your virtual environment ---

source ~/.bashrc
conda activate llama-env

### --- 5. Install/verify dependencies ---
echo "Installing/verifying dependencies for generative model..."

pip install -q --upgrade torch torchvision torchaudio
pip install -q transformers>=4.30.0
pip install -q scikit-learn pandas numpy tqdm

# Optional
pip install -q wandb matplotlib seaborn

echo "Dependencies verified successfully!"

### --- 6. Create necessary directories ---
mkdir -p logs
mkdir -p trained_models
mkdir -p $DATA_ROOT/generative_model

# Copy the generative model script to the correct location
cp generative_multimodal.py $DATA_ROOT/generative_model/

### --- 7. Run the generative training script ---
echo "Starting Generative Multi-Modal Training..."
echo "==========================================="

export PYTHONUNBUFFERED=1

# Change to the correct directory and run
cd $DATA_ROOT/generative_model
srun python generative_multimodal.py 2>&1 | tee ../logs/generative_training_$(date +%Y%m%d_%H%M%S).log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "============================================="
    echo "✅ Training completed successfully!"
    echo "💾 Check trained_models/ directory for saved models"
    echo "📋 Check logs/ directory for detailed training logs"
else
    echo "❌ Training failed with exit code ${PIPESTATUS[0]}"
    echo "📋 Check the log files for error details"
    exit 2
fi

echo "============================================="
echo "🎉 Job completed