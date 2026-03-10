#!/bin/bash
#SBATCH --job-name=geneformer_train         # Job name (can override at submit time with --job-name)
#SBATCH --output=%x_%j.out                  # Standard output file (%x=job name, %j=job id)
#SBATCH --error=%x_%j.err                   # Error log file (%x=job name, %j=job id)
#SBATCH --time=24:00:00                     # Walltime (can override: sbatch --time=12:00:00 script.sh)
#SBATCH --mem=256gb                         # RAM (can override: sbatch --mem=128gb script.sh)
#SBATCH --cpus-per-task=16                  # CPU cores
#SBATCH --partition=gpu                     # GPU partition
#SBATCH -G h100:1                           # Request 1 x H100 GPU (can override: sbatch -G h100:2 script.sh)

# ---------------------------------------------
# User Configuration - Edit these for your setup
# ---------------------------------------------
# Set environment defaults for base
CONDA_BASE="${CONDA_BASE:-/scratch/${USER}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-/scratch/${USER}/conda_envs/env_geneformer}"
GENEFORMER_WORKSPACE="${GENEFORMER_WORKSPACE:-/scratch/${USER}/Geneformer}"
BASE_DIR="${BASE_DIR:-/scratch/${USER}/geneformer_benchmarking}"


# Source a config file if it exists (recommended for custom settings)
CONFIG_FILE_BASE="${BASE_DIR}/scripts/slurm/config_myexp.sh"
CONFIG_FILE_WORKSPACE="${GENEFORMER_WORKSPACE}/scripts/slurm/config_myexp.sh"

if [ -f "$CONFIG_FILE_BASE" ]; then
    echo "Loading configuration from $CONFIG_FILE_BASE"
    source "$CONFIG_FILE_BASE"
elif [ -f "$CONFIG_FILE_WORKSPACE" ]; then
    echo "Loading configuration from $CONFIG_FILE_WORKSPACE"
    source "$CONFIG_FILE_WORKSPACE"
else
    echo "No custom config found in BASE_DIR or GENEFORMER_WORKSPACE; relying on defaults in the training python script unless env overrides are set"
fi

# ---------------------------------------------
# Set PyTorch CUDA allocator BEFORE any Python/torch loads
# ---------------------------------------------
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------
# Load and activate conda environment
# ---------------------------------------------
source ${CONDA_BASE}/etc/profile.d/conda.sh 
conda activate ${CONDA_ENV}

# ---------------------------------------------
# Optional: Use scratch for faster I/O
# ---------------------------------------------
SCRATCH_LABEL="${OUTPUT_PREFIX:-v1_10m_citeseq_pbmc_l3_cellclassification_fine_tuned_model_epoch5}"
SCRATCH_DIR="/scratch/$USER/${SCRATCH_LABEL}_$SLURM_JOB_ID"
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

# Copy the Python script and any required files from project folder
cp ${BASE_DIR}/scripts/slurm/train_geneformer_cellclassification.py .

# ---------------------------------------------
# Log environment (debug + reproducibility)
# ---------------------------------------------
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on node: $(hostname)"
echo "Workspace: ${GENEFORMER_WORKSPACE}"
echo "Model: ${MODEL_VERSION:-<python default>}"
echo "Number of layers to freeze: ${FREEZE_LAYERS:-<python default>}"
which python
python --version

# ---------------------------------------------
# Log GPU state for debugging
# ---------------------------------------------
echo "=== GPU status at job start ==="
nvidia-smi
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"

# ---------------------------------------------
# Run the Python script
# ---------------------------------------------
echo "Starting training at $(date)"
python train_geneformer_cellclassification.py
echo "Training finished at $(date)"



