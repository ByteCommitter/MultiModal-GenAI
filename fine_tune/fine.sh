(base) [dipanjan@hpc01 DIAC-WOZ]$ cat fine.sh
#!/bin/bash
#SBATCH --job-name=finetune-gpt2       # Name your job
#SBATCH --output=logs/job_%j.out       # Log output file
#SBATCH --partition=gpu_a100_8         # SLURM partition to run on
#SBATCH --nodelist=gpunode4             # Specific node (optional)
#SBATCH --gres=gpu:1                   # Number of GPUs
#SBATCH --time=10:00:00                # Max wall time
#SBATCH --mem=32G                     # Memory
#SBATCH --cpus-per-task=4             # CPU cores

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

### --- 2. Setup CUDA environment for bitsandbytes ---

# Only set CUDA_HOME if nvcc is available in PATH
if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi

export BITSANDBYTES_NOWELCOME=1

### --- 3. Activate your virtual environment ---

source ~/.bashrc
conda activate llama-env

### --- 4. Run your fine-tuning script ---

srun python fine_tune.py