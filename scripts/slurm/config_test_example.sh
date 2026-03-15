#!/bin/bash
# ---------------------------------------------
# Geneformer Testing Configuration Example
# ---------------------------------------------
# Copy this file and customize for your setup, then source it before running the SLURM script:
#   cp config_test_example.sh config_test_myexp.sh
#   # Edit config_test_myexp.sh with your paths
#   source config_test_myexp.sh
#   sbatch slurm_test_geneformer_cellclassification.sh

# ------------------------------------------------------------------------------------------------
# System Paths (required - customize for your HPC if different from defaults in the SLURM script)
# ------------------------------------------------------------------------------------------------
# export CONDA_BASE="/scratch/${USER}/miniconda3"
# export CONDA_ENV="/scratch/${USER}/conda_envs/env_geneformer"
# export GENEFORMER_WORKSPACE="/scratch/${USER}/Geneformer"
# export BASE_DIR="/scratch/${USER}/geneformer_benchmarking"

# ---------------------------------------------
# Model Configuration
# ---------------------------------------------
export MODEL_VERSION="V2-104M"  # Options: V1-10M, V2-104M, V2-316M, V2-104M_CLcancer

# ---------------------------------------------
# Experiment Naming
# ---------------------------------------------
export EXPERIMENT_PREFIX="citeseq_pbmc_cellclassification"
export PRETRAINED_DIR="v2_104m_citeseq_pbmc_l3_cellclassification_fine_tuned_model_epoch5"
export CLASSIFIER_DIR="v2_l3_citeseq_classifier_dir"  # Directory containing prepared datasets
export OUTPUT_PREFIX="v2_104m_l3_test_eval"  


# ---------------------------------------------
# Classifier Parameters
# ---------------------------------------------
export NPROC="16"         # Number of CPU cores to use
export FORWARD_BATCH_SIZE="64"  # Batch size for forward pass during evaluation
export STATE_KEY="cell_type_l3"  # Cell annotation column for classification (e.g., cell_type_l1, cell_type_l3)


# -------------------------------------------------------------------------------------------------------
# Example Usage FOR H100 GPU for Different Models (adjust `FORWARD_BATCH_SIZE` based on your GPU memory)
# -------------------------------------------------------------------------------------------------------
# For V2-104M:
#   export MODEL_VERSION="V2-104M"
#   export FORWARD_BATCH_SIZE="64"
#
# For V2-316M:
#   export MODEL_VERSION="V2-316M"
#   export FORWARD_BATCH_SIZE="64"
#
# For V1-10M:
#   export MODEL_VERSION="V1-10M"
#   export FORWARD_BATCH_SIZE="200"
