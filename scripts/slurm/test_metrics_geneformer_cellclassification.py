#!/usr/bin/env python3

import torch
from pathlib import Path
import os
import pickle
from datasets import load_from_disk
from transformers import AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from geneformer import Classifier

############################
#       USER CONFIG
#############################

# Workspace root - can be set via environment variable or defaults to current setup
WORKSPACE_ROOT = Path(os.environ.get("GENEFORMER_WORKSPACE", "/scratch/tmurugan/Geneformer"))

# Model and experiment configuration
MODEL_VERSION = os.environ.get("MODEL_VERSION", "V1-10M")  # Options: V1-10M, V2-104M, V2-104M_CLcancer, V2-316M
EXPERIMENT_PREFIX = os.environ.get("EXPERIMENT_PREFIX", "citeseq_pbmc_cellclassification")
PRETRAINED_DIR = Path(os.environ.get("PRETRAINED_DIR", "v1_10m_citeseq_pbmc_l3_cellclassification_fine_tuned_model_epoch5/260310_geneformer_cellClassifier_v1_10m_citeseq_pbmc_l3_cellclassification_fine_tuned_model_epoch5/ksplit1/checkpoint-34885"))
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "v1_10m_l3_test_eval")
CLASSIFIER_DIR = os.environ.get("CLASSIFIER_DIR", "v1_l3_citeseq_classifier_dir")  # Directory containing prepared datasets
BASE_DIR = Path(os.environ.get("BASE_DIR", "/scratch/tmurugan/geneformer_benchmarking"))


# tokenized + split data
TEST_PATH  = BASE_DIR / "data" / "classifier" / CLASSIFIER_DIR / f"{EXPERIMENT_PREFIX}_labeled_test.dataset"
ID_DICT_PATH = BASE_DIR / "data" / "classifier" / CLASSIFIER_DIR / f"{EXPERIMENT_PREFIX}_id_class_dict.pkl"

# fine-tuned Geneformer model directory
MODEL_DIR = BASE_DIR / "data" / "output" / PRETRAINED_DIR

# output directory where evaluation results will be saved
OUT_DIR = BASE_DIR / "data" / "output" / f"{OUTPUT_PREFIX}_{EXPERIMENT_PREFIX}"

# state key used during tokenization
STATE_KEY = os.environ.get("STATE_KEY", "cell_type_l3")

# General compute settings
NPROC = int(os.environ.get("NPROC", "16"))
FORWARD_BATCH_SIZE = int(os.environ.get("FORWARD_BATCH_SIZE", "200"))

# Derive model family (V1 or V2) from MODEL_VERSION for Classifier
# Extract "V1" or "V2" from strings like "V1-10M", "V2-104M", "V2-316M"
MODEL_FAMILY = MODEL_VERSION.split("-")[0] if "-" in MODEL_VERSION else MODEL_VERSION

#############################
#   LOAD DATASETS
#############################

print("Loading test dataset...")
test_ds = load_from_disk(TEST_PATH)
print("Loaded test set:", len(test_ds))

#############################
#   LOAD CLASS LABEL MAP
#############################

with open(ID_DICT_PATH, "rb") as f:
    id_class_dict = pickle.load(f)

#############################
#   DETERMINE CLASSES
#############################

if "label" not in test_ds.column_names:
    raise RuntimeError("Expected 'label' column missing from test dataset.")

num_classes = len(id_class_dict)
print("num_classes:", num_classes)

#############################
#       EMPTY CACHE
#############################

import gc, torch
gc.collect()
torch.cuda.empty_cache()

#############################
#   INITIALIZE CLASSIFIER
#############################

# Create a Classifier object ONLY for evaluation utilities
cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": STATE_KEY, "states": "all"},
    filter_data=None,
    training_args=None,   # <- no training args needed
    freeze_layers=0,
    num_crossval_splits=1,
    forward_batch_size=FORWARD_BATCH_SIZE,
    model_version=MODEL_FAMILY,  # Automatically derived from MODEL_VERSION (V1 or V2)
    nproc=NPROC,
)

#############################
#   RUN EVALUATION
#############################

print("\n=== Evaluating Model ===")
OUT_DIR.mkdir(parents=True, exist_ok=True)

metrics = cc.evaluate_saved_model(
    model_directory=MODEL_DIR,
    id_class_dict_file=ID_DICT_PATH,
    test_data_file=TEST_PATH,
    output_directory=OUT_DIR,
    output_prefix=OUTPUT_PREFIX,
    predict=True
)

print("Saved test metrics to:", OUT_DIR)
print("Accuracy:", metrics["acc"])
print("Macro F1:", metrics["macro_f1"])
print("Confusion matrix:", metrics["conf_matrix"])
metrics