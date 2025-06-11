from typing import Any, Literal

import os
import json

import torch
import torchvision

from nnopt.model.prune import l1_unstructured_pruning

import logging

from nnopt.model.const import (
    DEVICE, DTYPE, BASE_MODEL_DIR,
    CIFAR10_TRAIN_DIR, CIFAR10_TRAIN_PT_FILE,
    CIFAR10_VAL_DIR, CIFAR10_VAL_PT_FILE,
    CIFAR10_TEST_DIR, CIFAR10_TEST_PT_FILE,
    MOBILENETV2_CIFAR10_PT_FILENAME, METADATA_FILENAME
)


# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)
logger.info(f"Using device: {DEVICE}, dtype: {DTYPE}")

# MobileNetV2 model
def get_mobilenetv2_model(
    weights: torchvision.models.MobileNet_V2_Weights | None = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1,
    quantized: bool = False
    ) -> torch.nn.Module:
    """
    Returns the base MobileNetV2 model from torchvision.
    Args:
        quantized (bool): Whether to return a quantized version of the model.
    Returns:
        torch.nn.Module: The MobileNetV2 model.
    """
    logger.info(f"Loading base MobileNetV2 model, quantized: {quantized}")
    if not quantized:
        return torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        # Load the quantized MobileNetV2
        return torchvision.models.quantization.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1, quantize=False)

def get_mobilenetv2_cifar10_model(
    models_dir_path: str = BASE_MODEL_DIR,
    version: Literal["baseline"] = "baseline",
    quantized: bool = False
) -> tuple[torch.nn.Module, dict[str, Any] | None]:
    """
    Loads the MobileNetV2 model for CIFAR-10 from the specified directory and version.
    Args:
        models_dir_path (str): The base directory where the model is saved.
        version (str): The version of the model to load.
    Returns:
        torch.nn.Module: The MobileNetV2 model adapted for CIFAR-10.
    """
    # Loads the MobileNetV2 model for CIFAR-10, adapting the final layer for 10 classes.
    logger.info(f"Loading MobileNetV2 model for CIFAR-10 from version: {version} at {models_dir_path}")
    model = get_mobilenetv2_model(weights=None, quantized=quantized)
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
    version_path = os.path.join(version_dir_path, MOBILENETV2_CIFAR10_PT_FILENAME)
    if os.path.exists(models_dir_path) and os.path.exists(version_dir_path) and os.path.exists(version_path):
        model.load_state_dict(torch.load(version_path, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"Model state dictionary not found at {version_path}. Please ensure the model is saved correctly or the path and version are correct.")
    
    # Load metadata if it exists
    metadata: dict[str, Any] | None = None
    metadata_path = os.path.join(version_dir_path, METADATA_FILENAME)
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            logger.info(f"Loaded metadata: {metadata}")
            # If unstructured sparse config is present, apply it to reparametrize the model for further finetuning without touching to the pruned weights. This works because as pruned weights are already set to 0, pruning again with the same config will only prune the already pruned model weights.
            if "unstructured_sparse_config" in metadata:
                unstruct_sparse_config = metadata["unstructured_sparse_config"]
                if unstruct_sparse_config:
                    logger.info(f"Applying unstructured sparse config: {unstruct_sparse_config}")
                    model = l1_unstructured_pruning(model, **unstruct_sparse_config)
    else:
        logger.warning(f"No metadata found at {metadata_path}. Continuing without applying unstructured sparse config.")
    return model, metadata


def save_mobilenetv2_cifar10_model(
    model: torch.nn.Module,
    version: str,
    unstruct_sparse_config: dict[str, Any] = None,
    metrics_values: dict[str, Any] | None = None,
    models_dir_path: str = BASE_MODEL_DIR,
) -> None:
    """
    Saves the MobileNetV2 model for CIFAR-10 to the specified directory.
    Args:
        model (torch.nn.Module): The MobileNetV2 model to save.
        models_dir_path (str): The base directory where the model will be saved.
        version (str): The version of the model to save.
    """
    # Calculate the paths
    version_dir_path = os.path.join(models_dir_path, version)
    if not os.path.exists(version_dir_path):
        os.makedirs(version_dir_path)
    model_path = os.path.join(version_dir_path, MOBILENETV2_CIFAR10_PT_FILENAME)
    metadata_path = os.path.join(version_dir_path, METADATA_FILENAME)
    # Save the model state dictionary
    torch.save(model.state_dict(), model_path)
    # Save metadata if provided
    metadata: dict[str, Any] = {}
    if unstruct_sparse_config is not None:
        metadata["unstructured_sparse_config"] = unstruct_sparse_config
    if metrics_values is not None:
        metadata["metrics_values"] = metrics_values
    if metadata:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            logger.info(f"Metadata saved to {metadata_path}")
    else:
        logger.info("No metadata to save.")
    # Save the model
    logger.info(f"Model saved to {model_path}")


# CIFAR-10 dataset transforms for MobileNetV2
def get_mobilenetv2_cifar10_transforms(
    color_jitter: bool = True
):
    """
    Returns the transforms for training and validation datasets for MobileNetV2 on CIFAR-10.
    Args:
        color_jitter (bool): Whether to apply color jitter augmentation.
    Returns:
        tuple: A tuple containing the training and validation transforms.
    """
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) if color_jitter else torchvision.transforms.Lambda(lambda x: x),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # Use ImageNet means/stds
    ])
    
    eval_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # Use ImageNet means/stds
    ])
    
    return train_transform, eval_transform


# CIFAR-10 datasets
def get_cifar10_datasets(
    color_jitter: bool = True
):
    """
    Returns the CIFAR-10 datasets with transforms for MobileNetV2.
    Args:
        color_jitter (bool): Whether to apply color jitter augmentation.
    Returns:
        tuple: A tuple containing the training, validation, and test datasets.
    """
    image_train_cifar10_mobilenetV2_transform, image_eval_cifar10_mobilenetV2_transform = get_mobilenetv2_cifar10_transforms(color_jitter=color_jitter)

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
        val_dataset.dataset.transform = image_eval_cifar10_mobilenetV2_transform  # Ensure validation dataset has the correct transform
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
            transform=image_eval_cifar10_mobilenetV2_transform,
            download=True
        )
        torch.save(test_dataset, CIFAR10_TEST_PT_FILE)
    else:
        logger.info("Loading existing test dataset...")
        test_dataset = torch.load(CIFAR10_TEST_PT_FILE, weights_only=False)

    return train_dataset, val_dataset, test_dataset