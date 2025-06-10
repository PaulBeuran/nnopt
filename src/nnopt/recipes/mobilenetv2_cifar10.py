from typing import Literal

import os

import torch
import torchvision

import logging

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
MOBILENETV2_CIFAR10_BASELINE_PT_FILENAME = "mobilenetv2_cifar10.pt"
MOBILENETV2_CIFAR10_BASELINE_PT_FILE = os.path.join(MODEL_BASELINE_DIR, MOBILENETV2_CIFAR10_BASELINE_PT_FILENAME)

# Setup logging
logging.basicConfig(level=logging.DEBUG, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)
logger.debug(f"Using device: {DEVICE}, dtype: {DTYPE}")

# MobileNetV2 model
def get_mobilenetv2_cifar10_model(
    models_dir_path: str = BASE_MODEL_DIR,
    version: Literal["baseline"] = "baseline",
) -> torch.nn.Module:
    # Loads the MobileNetV2 model for CIFAR-10, adapting the final layer for 10 classes.
    logger.info(f"Loading MobileNetV2 model for CIFAR-10 from version: {version} at {models_dir_path}")
    model = torchvision.models.mobilenet_v2(weights=None)
    num_classes_cifar10 = 10
    if hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Sequential):
        if hasattr(model.classifier[-1], 'in_features'):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_features, num_classes_cifar10)
        else:
            # This case should ideally not be hit if MobileNetV2 structure is standard
            logger.error("Could not find 'in_features' in the last layer of the classifier to adapt.")
            raise AttributeError("Could not find 'in_features' in the last layer of the classifier.")
    elif hasattr(model, 'fc'): # Fallback for models using 'fc'
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes_cifar10)
    else:
        logger.error("Model does not have a known 'classifier' (Sequential) or 'fc' (Linear) attribute to adapt.")
        raise AttributeError("Model does not have a known classifier structure to adapt.")
    
    # Load the model state dictionary from the specified version
    version_dir_path = os.path.join(models_dir_path, version)
    version_path = os.path.join(version_dir_path, MOBILENETV2_CIFAR10_BASELINE_PT_FILENAME)
    if os.path.exists(models_dir_path) and os.path.exists(version_dir_path) and os.path.exists(version_path):
        model.load_state_dict(torch.load(version_path, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"Model state dictionary not found at {version_path}. Please ensure the model is saved correctly or the path and version are correct.")
    return model

def save_mobilenetv2_cifar10_model(
    model: torch.nn.Module,
    models_dir_path: str = BASE_MODEL_DIR,
    version: Literal["baseline"] = "baseline",
) -> None:
    """
    Saves the MobileNetV2 model for CIFAR-10 to the specified directory.
    Args:
        model (torch.nn.Module): The MobileNetV2 model to save.
        models_dir_path (str): The base directory where the model will be saved.
        version (str): The version of the model to save.
    """
    version_dir_path = os.path.join(models_dir_path, version)
    if not os.path.exists(version_dir_path):
        os.makedirs(version_dir_path)
    version_path = os.path.join(version_dir_path, MOBILENETV2_CIFAR10_BASELINE_PT_FILENAME)
    torch.save(model.state_dict(), version_path)
    logger.info(f"Model saved to {version_path}")

# CIFAR-10 dataset transforms for MobileNetV2
def get_mobilenetv2_cifar10_transforms(
    color_jitter: bool = True
):
    """
    Returns the transforms for training and validation datasets for MobileNetV2 on CIFAR-10.
    """
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) if color_jitter else torchvision.transforms.Lambda(lambda x: x),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # Use ImageNet means/stds
    ])
    
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # Use ImageNet means/stds
    ])
    
    return train_transform, val_transform

# CIFAR-10 datasets
def get_cifar10_datasets(
    color_jitter: bool = True
):
    """
    Returns the CIFAR-10 datasets with transforms for MobileNetV2.
    """
    image_train_cifar10_mobilenetV2_transform, image_val_cifar10_mobilenetV2_transform = get_mobilenetv2_cifar10_transforms(color_jitter=color_jitter)

    if (not os.path.exists(CIFAR10_TEST_PT_FILE) or
        not os.path.exists(CIFAR10_VAL_PT_FILE)):
        logger.info("Training and/or validation dataset does not exist, creating, splitting and saving...")
        train_dataset = torchvision.datasets.CIFAR10(
            root=CIFAR10_TRAIN_DIR,
            train=True,
            transform=image_train_cifar10_mobilenetV2_transform,
            download=True
        )
        train_val_split_generator = torch.Generator().manual_seed(42069)  # For reproducibility
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [45000, 5000], 
                                                                generator=train_val_split_generator)
        torch.save(train_dataset, CIFAR10_TRAIN_PT_FILE)
        torch.save(val_dataset, CIFAR10_VAL_PT_FILE)
    else:
        logger.info("Loading existing training and validation datasets...")
        train_dataset = torch.load(CIFAR10_TRAIN_PT_FILE, weights_only=False)
        val_dataset = torch.load(CIFAR10_VAL_PT_FILE, weights_only=False)
    if not os.path.exists(CIFAR10_TEST_PT_FILE):
        logger.info("Test dataset does not exist, creating and saving...")
        test_dataset = torchvision.datasets.CIFAR10(
            root=CIFAR10_TEST_DIR,
            train=False,
            transform=image_val_cifar10_mobilenetV2_transform,
            download=True
        )
        torch.save(test_dataset, CIFAR10_TEST_PT_FILE)
    else:
        logger.info("Loading existing test dataset...")
        test_dataset = torch.load(CIFAR10_TEST_PT_FILE, weights_only=False)

    return train_dataset, val_dataset, test_dataset