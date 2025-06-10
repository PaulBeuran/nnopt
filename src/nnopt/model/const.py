import os

import torch

# Setup device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Base directories for datasets and models
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_CURRENT_DIR)))
BASE_DATA_DIR = os.path.join(_WORKSPACE_DIR, "data")
IMAGE_DATA_DIR = os.path.join(BASE_DATA_DIR, "image")
BASE_MODEL_DIR = os.path.join(_WORKSPACE_DIR, "models")
MODEL_BASELINE_DIR = os.path.join(BASE_MODEL_DIR, "baseline")

# CIFAR-10 dataset directories
CIFAR10_DIR = os.path.join(IMAGE_DATA_DIR, "cifar10")
CIFAR10_TRAIN_DIR = os.path.join(CIFAR10_DIR, "train")
CIFAR10_TRAIN_PT_FILE = os.path.join(CIFAR10_TRAIN_DIR, "data.pt")
CIFAR10_VAL_DIR = os.path.join(CIFAR10_DIR, "val")
CIFAR10_VAL_PT_FILE = os.path.join(CIFAR10_VAL_DIR, "data.pt")
CIFAR10_TEST_DIR = os.path.join(CIFAR10_DIR, "test")
CIFAR10_TEST_PT_FILE = os.path.join(CIFAR10_TEST_DIR, "data.pt")

# MobileNetV2 model directory
MOBILENETV2_CIFAR10_PT_FILENAME = "model.pt"
METADATA_FILENAME = "metadata.json"
