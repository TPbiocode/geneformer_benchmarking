#!/bin/bash
# ---------------------------------------------
# Geneformer Training Configuration Example
# ---------------------------------------------
# Copy this file and customize for your setup, then source it before running the SLURM script:
#   cp config_example.sh config_myexp.sh
#   # Edit config_myexp.sh with your paths
#   source config_myexp.sh
#   sbatch citeseq_pbmc_classification_l3.sh

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
export EXPERIMENT_PREFIX="v2_l3_citeseq_pbmc_cellclassification"
export CLASSIFIER_DIR="v2_l3_citeseq_classifier_dir"  # Directory containing prepared datasets
export OUTPUT_PREFIX="v2_104m_freeze4_citeseq_pbmc_l3_epoch5"


# ---------------------------------------------
# Training Parameters
# ---------------------------------------------
export FREEZE_LAYERS="4"  # Number of layers to freeze (proportional: V1=2, V2-104M=4, V2-316M=6)
export NPROC="16"         # Number of CPU cores to use
export BATCH_SIZE="16"      # Batch size for training
export EPOCHS="5"          # Number of training epochs
export STATE_KEY="cell_type_l3"  # Cell annotation column for classification (e.g., cell_type_l1, cell_type_l3)


# ---------------------------------------------
# Example Usage for Different Models
# ---------------------------------------------
# For V2-104M with proportional freezing (67% trainable):
#   export MODEL_VERSION="V2-104M"
#   export FREEZE_LAYERS="4"
#   export OUTPUT_SUFFIX="v2_104m_freeze4_l3_epoch5"
#
# For V2-316M with proportional freezing (67% trainable):
#   export MODEL_VERSION="V2-316M"
#   export FREEZE_LAYERS="6"
#   export OUTPUT_SUFFIX="v2_316m_freeze6_l3_epoch5"
#
# For V1-10M (baseline):
#   export MODEL_VERSION="V1-10M"
#   export FREEZE_LAYERS="2"
#   export OUTPUT_SUFFIX="v1_10m_freeze2_l3_epoch5"
