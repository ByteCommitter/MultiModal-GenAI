#!/bin/bash
#
#SBATCH --job-name=genai_semantic
#SBATCH --output=/home/dipanjan/rugraj/DIAC-WOZ/genai_semantic_%j.out
#SBATCH --error=/home/dipanjan/rugraj/DIAC-WOZ/genai_semantic_%j.err
#SBATCH --partition=gpu_a100_8        # change to gpu_h100_4 or other available partition if needed
#SBATCH --gres=gpu:1                  # number of GPUs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# -----------------------
# User-editable settings
# -----------------------
# Full path to the python script on the cluster (edit if different)
PY_SCRIPT="/home/dipanjan/rugraj/DIAC-WOZ/generate_semantic_features.py"
# Conda environment name that has torch/transformers installed
CONDA_ENV="llama-env"
# Data directory (where metadata and outputs live)
DATA_DIR="/home/dipanjan/rugraj/DIAC-WOZ"
# Local HF model cache base (usually under $HOME/.cache/huggingface)
HF_HOME="${HOME}/.cache/huggingface"
# How often to sync logs (seconds)
SYNC_INTERVAL=10

# -----------------------
# Optional module loads
# -----------------------
# If your cluster uses modules, uncomment relevant lines (adjust module names)
# module purge
# module load cuda/11.8
# module load cudnn

# -----------------------
# Environment setup (robust conda init)
# -----------------------
set -o pipefail
set -u

echo "Job starting on $(hostname) at $(date)"
echo "Using script: ${PY_SCRIPT}"
echo "Data dir: ${DATA_DIR}"
echo "Conda env: ${CONDA_ENV}"

# Prefer user's Miniconda/Anaconda conda executable if present
CONDA_CANDIDATES=(
    "${HOME}/miniconda3/bin/conda"
    "${HOME}/anaconda3/bin/conda"
    "${HOME}/.conda/bin/conda"
    "/opt/conda/bin/conda"
    "$(command -v conda 2>/dev/null || true)"
)
CONDA_EXEC=""
for c in "${CONDA_CANDIDATES[@]}"; do
    if [ -n "${c}" ] && [ -x "${c}" ]; then
        CONDA_EXEC="${c}"
        break
    fi
done

if [ -z "${CONDA_EXEC}" ]; then
    echo "WARNING: No conda executable found in common locations or PATH. 'conda run' will fallback to system conda if available."
    CONDA_EXEC="$(command -v conda 2>/dev/null || true)"
fi

echo "Using conda executable: ${CONDA_EXEC}"
if [ -n "${CONDA_EXEC}" ]; then
    "${CONDA_EXEC}" --version || true
fi

# Force Python to ignore user site-packages (prevents ~/.local packages from shadowing env packages)
export PYTHONNOUSERSITE=1

# Export HF/cache env vars
export HF_HOME="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}

# -------------------------
# Pre-flight checks using the selected conda exec
# -------------------------
echo "--- Pre-flight checks (using selected conda) ---"
if [ -n "${CONDA_EXEC}" ]; then
    # Use conda run via the selected conda executable to ensure the chosen env's python is used.
    "${CONDA_EXEC}" run -n "${CONDA_ENV}" --no-capture-output python - <<'PYCODE' || true
import sys, os
print("python executable:", sys.executable)
print("python version:", sys.version.splitlines()[0])
# torch check
try:
    import torch
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("CUDA device name:", torch.cuda.get_device_name(0))
        except Exception:
            pass
except Exception as e:
    print("torch import/check failed:", e)
# transformers check
try:
    import transformers
    print("transformers version:", transformers.__version__)
except Exception as e:
    print("transformers import failed:", e)
# markupsafe diagnostic
try:
    import markupsafe
    print("markupsafe:", getattr(markupsafe, "__version__", None), "file:", getattr(markupsafe, "__file__", None))
except Exception as me:
    print("markupsafe import failed:", me)
print("HF model cache exists:", os.path.exists(os.path.expanduser("~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf")))
PYCODE
else
    # Fallback: run checks with plain python (may be system python)
    python - <<'PYCODE' || true
import sys, os
print("No explicit conda exec; python executable:", sys.executable)
print("python version:", sys.version.splitlines()[0])
print("HF model cache exists:", os.path.exists(os.path.expanduser("~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf")))
PYCODE
fi
echo "-------------------------"

# Ensure script exists
if [ ! -f "${PY_SCRIPT}" ]; then
    echo "ERROR: Python script not found at ${PY_SCRIPT}" >&2
    exit 1
fi

# Run the job (unbuffered) using the selected conda exec
if [ -n "${CONDA_EXEC}" ]; then
    echo "Running with: ${CONDA_EXEC} run -n ${CONDA_ENV} --no-capture-output python -u ${PY_SCRIPT} --metadata_csv ${DATA_DIR}/daic_metadata.csv --output_dir ${DATA_DIR}"
    srun "${CONDA_EXEC}" run -n "${CONDA_ENV}" --no-capture-output python -u "${PY_SCRIPT}" --metadata_csv "${DATA_DIR}/daic_metadata.csv" --output_dir "${DATA_DIR}"
else
    echo "Running fallback: python -u ${PY_SCRIPT} --metadata_csv ${DATA_DIR}/daic_metadata.csv --output_dir ${DATA_DIR}"
    srun python -u "${PY_SCRIPT}" --metadata_csv "${DATA_DIR}/daic_metadata.csv" --output_dir "${DATA_DIR}"
fi

echo "Job finished at $(date)"
