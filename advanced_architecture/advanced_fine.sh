#!/bin/bash
#SBATCH --job-name=generative-multimodal  # Changed job name
#SBATCH --output=logs/generative_job_%j.out
#SBATCH --partition=gpu_a100_8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00                   # Increased time for diffusion training
#SBATCH --mem=96G                         # Increased memory for larger model
#SBATCH --cpus-per-task=12                # Increased CPUs for data processing

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

# Only set CUDA_HOME if nvcc is available in PATH
if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export PATH=$CUDA_HOME/bin:$PATH
fi

# Additional environment variables for diffusion training
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1
export WANDB_MODE=offline  # Optional: disable wandb if not needed

# Set environment variables for better performance
export TORCH_CUDA_ARCH_LIST="8.0"  # For A100 GPUs
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings

### --- 3. Print system information ---
echo "=== System Information ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
echo "CUDA Version: $(nvcc --version | grep release)"
echo "Python Version: $(python --version)"
echo "=========================="

### --- 4. Activate your virtual environment ---

source ~/.bashrc
conda activate llama-env

### --- 5. Install additional dependencies if needed ---
echo "Installing/updating dependencies..."

# Install required packages for generative model
pip install -q --upgrade torch torchvision torchaudio
pip install -q transformers>=4.30.0
pip install -q scikit-learn pandas numpy tqdm
pip install -q einops  # Required for diffusion operations
pip install -q wandb
pip install -q pytorch-lightning  # Optional: for structured training

echo "Dependencies installed successfully!"

### --- 6. Create logs directory ---
mkdir -p logs

### --- 7. Run your generative fine-tuning script ---
echo "Starting Generative Multi-Modal Training..."
echo "=========================================================="

# Set additional training parameters
export DIFFUSION_STEPS=1000
export GUIDANCE_SCALE=3.0
export BATCH_SIZE=4  # Reduced batch size for larger model

# Run with detailed logging
srun python advanced_multimodal.py \
    --model_type generative \
    --num_timesteps $DIFFUSION_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --batch_size $BATCH_SIZE \
    2>&1 | tee logs/generative_training_$(date +%Y%m%d_%H%M%S).log

echo "=========================================================="
echo "Training completed at: $(date)"
echo "Check logs/ directory for detailed training logs"
