#!/usr/bin/env python3

import anndata
import os
from pathlib import Path
import pickle
import urllib.request
from datasets import load_from_disk
import zipfile
from geneformer import Classifier
from transformers import TrainingArguments

#############################
#       USER CONFIG
#############################

# Workspace root - can be set via environment variable or defaults to current setup
WORKSPACE_ROOT = Path(os.environ.get("GENEFORMER_WORKSPACE", "/scratch/tmurugan/Geneformer"))

# Model and experiment configuration
MODEL_VERSION = os.environ.get("MODEL_VERSION", "V1-10M")  # Options: V1-10M, V2-104M, V2-104M_CLcancer, V2-316M
EXPERIMENT_PREFIX = os.environ.get("EXPERIMENT_PREFIX", "citeseq_pbmc_cellclassification")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "v1_10m_citeseq_pbmc_l3_cellclassification_fine_tuned_model_epoch5")
CLASSIFIER_DIR = os.environ.get("CLASSIFIER_DIR", "v1_l3_citeseq_classifier_dir")  # Directory containing prepared datasets
BASE_DIR = Path(os.environ.get("BASE_DIR", "/scratch/tmurugan/geneformer_benchmarking"))


# tokenized + split data
TRAIN_PATH = BASE_DIR / "data" / "classifier" / CLASSIFIER_DIR / f"{EXPERIMENT_PREFIX}_labeled_train.dataset"
TEST_PATH  = BASE_DIR / "data" / "classifier" / CLASSIFIER_DIR / f"{EXPERIMENT_PREFIX}_labeled_test.dataset"
ID_DICT_PATH = BASE_DIR / "data" / "classifier" / CLASSIFIER_DIR / f"{EXPERIMENT_PREFIX}_id_class_dict.pkl"

# pretrained Geneformer model directory
MODEL_DIR = WORKSPACE_ROOT / f"Geneformer-{MODEL_VERSION}"

# output directory where fine-tuned model will be saved
OUT_MODEL_DIR = BASE_DIR / "data" / "output" / OUTPUT_PREFIX

# state key used during tokenization
STATE_KEY = os.environ.get("STATE_KEY", "cell_type_l3")

# General compute settings
NPROC = int(os.environ.get("NPROC", "16"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
EPOCHS = int(os.environ.get("EPOCHS", "5"))
FREEZE_LAYERS = int(os.environ.get("FREEZE_LAYERS", "2"))

# Derive model family (V1 or V2) from MODEL_VERSION for Classifier
# Extract "V1" or "V2" from strings like "V1-10M", "V2-104M", "V2-316M"
MODEL_FAMILY = MODEL_VERSION.split("-")[0] if "-" in MODEL_VERSION else MODEL_VERSION

#############################
#   LOAD DATASETS
#############################

print("Loading datasets...")
train_ds = load_from_disk(str(TRAIN_PATH))
test_ds  = load_from_disk(str(TEST_PATH))

############################
#   CREATE TRAIN/VAL SPLIT
############################

import numpy as np

ids = np.array(train_ds["join_id"])
splits = np.array(train_ds["split"])

train_ids = ids[splits == "train"]
eval_ids  = ids[splits == "validation"]

train_valid_id_split_dict = {
    "attr_key": "join_id",
    "train": train_ids.tolist(),
    "eval": eval_ids.tolist()
}

print(f"Train      : {len(train_ids)} cells")
print(f"Validation :  {len(eval_ids)} cells")
print(f"Test       :  {len(test_ds)} cells")

#############################
#   DETERMINE CLASSES
#############################

if "label" not in train_ds.column_names:
    raise RuntimeError("Expected 'label' column missing from train dataset.")

num_classes = len(set(train_ds["label"]))
print("num_classes =", num_classes)

#############################
#   LOAD CLASS LABEL MAP
#############################

with open(ID_DICT_PATH, "rb") as f:
    id_class_dict = pickle.load(f)


#############################
#   TRAINING ARGUMENTS
#############################

training_args = TrainingArguments(
    output_dir=str(OUT_MODEL_DIR),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=5e-5,      # default in training arguments
    num_train_epochs=EPOCHS, 
    lr_scheduler_type="linear", #earlier 'polynomial
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=False, # set to False to disable mixed precision training (default)
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=str(OUT_MODEL_DIR / "logs"),
    seed=73, #seed set for reproducibility
)

from inspect import signature

# get the raw dictionary of arguments from the TrainingArguments instance
raw_args_dict = vars(training_args)  
# compute allowed keys from TrainingArguments.__init__
sig = signature(TrainingArguments.__init__)
allowed = {name for name, param in sig.parameters.items() if name not in ("self", "kwargs")}

# find unexpected keys (debugging aid)
unexpected = set(raw_args_dict) - allowed
if unexpected:
    print("Filtering out unexpected TrainingArguments keys:", unexpected)

clean_args = {k: v for k, v in raw_args_dict.items() if k in allowed}

training_args_dict = clean_args


#############################
#   INITIALIZE CLASSIFIER
#############################

cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": STATE_KEY, "states": "all"},
    filter_data=None,
    training_args=training_args_dict,
    freeze_layers=FREEZE_LAYERS,
    num_crossval_splits=1,
    forward_batch_size=200,
    model_version=MODEL_FAMILY,  # Automatically derived from MODEL_VERSION (V1 or V2)
    nproc=NPROC,
)

#############################
#       EMPTY CACHE
#############################
import gc, torch
gc.collect()
torch.cuda.empty_cache()

#############################
#   TRAIN MODEL
#############################

print("\n=== Training Model ===")
OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

trainer = cc.validate(
    model_directory=MODEL_DIR,
    prepared_input_data_file=str(TRAIN_PATH),
    id_class_dict_file=str(ID_DICT_PATH),
    output_directory=str(OUT_MODEL_DIR),
    output_prefix=OUTPUT_PREFIX,
    split_id_dict=train_valid_id_split_dict,
    )

print("Training complete.")
print("Model saved to:", OUT_MODEL_DIR)