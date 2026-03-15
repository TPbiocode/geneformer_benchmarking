# Geneformer Benchmarking

This repository contains reproducible benchmarking scripts for Geneformer-based single-cell RNA-seq cell type classification, including donor-level splits, tokenization, model training, and evaluation for comparing Geneformer models (V1-10M, V2-104M, V2-316M).        

## Overview
This repository contains:
- Configurable SLURM training scripts for HPC environments
- Jupyter notebooks for data preprocessing and results analysis
- Utilities for fair model comparison with proportional layer freezing

## Requirements
- Geneformer package from [HuggingFace](https://huggingface.co/ctheodoris/Geneformer)
- Python 3.11+
- See `requirements.txt` for dependencies

## Usage
See `scripts/slurm/README.md` for detailed instructions.
