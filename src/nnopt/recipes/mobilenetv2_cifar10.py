from typing import Any, Literal
from collections import OrderedDict

import os
import json

import torch
import torchvision

from nnopt.model.prune import l1_unstructured_pruning, remove_pruning_reparameterization

import logging

from nnopt.model.const import (
    DEVICE, DTYPE, BASE_MODEL_DIR,
    CIFAR10_TRAIN_DIR, CIFAR10_TRAIN_PT_FILE,
    CIFAR10_VAL_DIR, CIFAR10_VAL_PT_FILE,
    CIFAR10_TEST_DIR, CIFAR10_TEST_PT_FILE, METADATA_FILENAME
)

# Define filenames for architecture and state_dict.
# TODO: These should ideally be moved to nnopt.model.const
TORCH_MODEL_PT_FILENAME = "model.pt"
TORCH_STATE_DICT_PT_FILENAME = "state_dict.pt"
JIT_SCRIPT_PT_FILENAME = "jit_script.pt"
JIT_TRACE_PT_FILENAME = "jit_trace.pt"

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)
logger.info(f"Using device: {DEVICE}, dtype: {DTYPE}")


def _replace_head(model: torch.nn.Module, num_classes: int, is_quantized: bool) -> torch.nn.Module:
    """
    Replaces the head of the model to match the number of classes.
    Args:
        model (torch.nn.Module): The model whose head needs to be replaced.
        num_classes (int): The number of output classes for the classifier.
    Returns:
        torch.nn.Module: The model with the modified head.
    """
    if hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Sequential):
        if hasattr(model.classifier[-1], 'in_features'):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_features, num_classes) if not is_quantized else torch.nn.quantized.Linear(in_features, num_classes)
        else:
            raise ValueError("The last layer of the classifier does not have 'in_features' attribute. Cannot adapt the model for CIFAR-10 classes.")
    elif hasattr(model, 'fc'):  # Fallback for models using 'fc'
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes) if not is_quantized else torch.nn.quantized.Linear(in_features, num_classes)
    else:
        raise ValueError("The model does not have a classifier or fc attribute. Cannot adapt the model for CIFAR-10 classes.")
    return model


# MobileNetV2 model
def init_mobilenetv2_cifar10_model(
    weights: torchvision.models.MobileNet_V2_Weights | None = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1,
    to_quantize: bool = False,
    is_quantized: bool = False,
    num_classes: int = 10
) -> torch.nn.Module:
    """
    Returns the base MobileNetV2 model from torchvision.
    Args:
        quantized (bool): Whether to return a quantized version of the model.
        quant_weights (bool): For quantized models, this is passed as the 'quantize' argument to torchvision's quantized mobilenet_v2.
                              If True, returns a model with quantized layer structure.
        num_classes (int): Number of output classes for the classifier.
    Returns:
        torch.nn.Module: The MobileNetV2 model.
    """
    logger.info(f"Loading MobileNetV2 model with weights: {weights}, to_quantize: {to_quantize}, is_quantized: {is_quantized}")
    if not to_quantize and not is_quantized:
        model = torchvision.models.mobilenet_v2(weights=weights)
    elif to_quantize or is_quantized:
        model = torchvision.models.quantization.mobilenet_v2(weights=weights, quantize=is_quantized)
    logger.info(f"Replacing head of the model to match {num_classes} classes")
    model = _replace_head(model, num_classes, is_quantized)
    return model


def save_mobilenetv2_cifar10_model(
    model: torch.nn.Module,
    version: str,
    unstruct_sparse_config: dict[str, Any] = None,
    metrics_values: dict[str, Any] | None = None,
    models_dir_path: str = BASE_MODEL_DIR,
    save_jit: bool = True
) -> None:
    """
    Saves the MobileNetV2 model architecture and state_dict for CIFAR-10 to the specified directory.
    Args:
        model (torch.nn.Module): The MobileNetV2 model to save.
        models_dir_path (str): The base directory where the model will be saved.
        version (str): The version of the model to save.
        unstruct_sparse_config (dict[str, Any]): Configuration for unstructured sparsity.
        metrics_values (dict[str, Any] | None): Metrics to save alongside the model.
    """
    # Calculate the paths
    version_dir_path = os.path.join(models_dir_path, version)
    if not os.path.exists(version_dir_path):
        os.makedirs(version_dir_path)

    # model_path and metadata_path will now use the global constants
    model_path = os.path.join(version_dir_path, TORCH_MODEL_PT_FILENAME)
    metadata_path = os.path.join(version_dir_path, METADATA_FILENAME)

    # Save the model
    torch.save(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save the state dictionary
    model_state_dict_path = os.path.join(version_dir_path, TORCH_STATE_DICT_PT_FILENAME)
    torch.save(model.state_dict(), model_state_dict_path)
    logger.info(f"Model state_dict saved to {model_state_dict_path}")

    # Save metadata if provided
    metadata: dict[str, Any] = {}
    if unstruct_sparse_config is not None:
        metadata["unstructured_sparse_config"] = unstruct_sparse_config
    if metrics_values is not None:
        metadata["metrics_values"] = metrics_values

    if metadata: # Ensure metadata is not empty before writing
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            logger.info(f"Metadata saved to {metadata_path}")
    else:
        logger.info("No metadata to save (metadata dictionary is empty).")

    # Remove the reparameterization for unstructured sparsity if it exists
    model = remove_pruning_reparameterization(model)

    # Save the model in JIT format if requested
    if save_jit:
        logger.info("Saving model in JIT script format...")
        model_script = torch.jit.script(model)
        model_script_path = os.path.join(version_dir_path, JIT_SCRIPT_PT_FILENAME)
        torch.jit.save(model_script, model_script_path)
        logger.info(f"JIT script model saved to {model_script_path}")
        logger.info("Saving model in JIT trace format...")
        model_trace = torch.jit.trace(model.cpu(), example_inputs=torch.randn(1, 3, 224, 224))
        model_trace_path = os.path.join(version_dir_path, JIT_TRACE_PT_FILENAME)
        torch.jit.save(model_trace, model_trace_path)
        logger.info(f"JIT model saved to {model_trace_path}")


def convert_mobilenetv2_cifar10_to_quantized(
    model: torch.nn.Module,
) -> torch.nn.Module:
    quant_model = init_mobilenetv2_cifar10_model(
        weights=None,      # No pretrained torchvision weights, we're loading our custom QAT model
        to_quantize=True,    # We want a quantized model architecture
        is_quantized=False, # This ensures 'quantize=True' is passed to torchvision's quantized constructor
        num_classes=10    # CIFAR-10 has 10 classes
    )
    # Change the head of the quantized model to match CIFAR-10 classes
    quant_model.load_state_dict(model.state_dict())
    quant_model.cpu()
    return quant_model


def load_mobilenetv2_cifar10_model(
    version: str,
    models_dir_path: str = BASE_MODEL_DIR,
    device: Literal["cpu", "cuda"] = DEVICE,
    mode: Literal["model", "state_dict", "state_load", "quant_state_load", "jit_script", "jit_trace"] = "model",
    convert_to_quantized: bool = False,
    restore_unstruct_prune_reparam: bool = False
) -> tuple[torch.nn.Module | OrderedDict | torch.jit.ScriptModule, dict[str, Any] | None]:
    """
    Loads the MobileNetV2 model for CIFAR-10 from the specified directory and version.
    It first loads the model architecture, then loads and applies the state_dict.
    Args:
        models_dir_path (str): The base directory where the model is saved.
        version (str): The version of the model to load.
        device (str): The device to load the model onto.
    Returns:
        tuple[torch.nn.Module, dict[str, Any] | None]: The MobileNetV2 model and its metadata.
    """
    logger.info(f"Loading MobileNetV2 model for CIFAR-10 from version: {version} at {models_dir_path}")
    
    # Define the paths for model, state_dict, metadata, and JIT trace
    version_dir_path = os.path.join(models_dir_path, version)
    model_path = os.path.join(version_dir_path, TORCH_MODEL_PT_FILENAME)
    state_dict_path = os.path.join(version_dir_path, TORCH_STATE_DICT_PT_FILENAME)
    metadata_path = os.path.join(version_dir_path, METADATA_FILENAME)
    jit_script_path = os.path.join(version_dir_path, JIT_SCRIPT_PT_FILENAME)
    jit_trace_path = os.path.join(version_dir_path, JIT_TRACE_PT_FILENAME)
    
    # Check and load the metadata if it exists
    model: torch.nn.Module | OrderedDict | torch.jit.ScriptModule = None
    metadata: dict[str, Any] | None = None
    if not os.path.exists(metadata_path):
        logger.warning(f"No metadata found at {metadata_path}. Continuing without metadata.")
    else:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            logger.info(f"Loaded metadata: {metadata}")

    # If mode is 'model', load the model from its file directly
    if mode == "model":
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model is saved correctly or the path and version are correct.")
        else:
            logger.info(f"Loading model from {model_path}")
            model: torch.nn.Module = torch.load(model_path, map_location=device, weights_only=False)

    elif mode == "state_dict":
        # Load the state_dict from the specified path
        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"State dictionary file not found at {state_dict_path}. Please ensure the model is saved correctly or the path and version are correct.")
        else:
            logger.info(f"Loading state_dict from {state_dict_path}")
            state_dict: OrderedDict = torch.load(state_dict_path, map_location=device, weights_only=True)
            logger.info(f"Successfully loaded state_dict from {state_dict_path}")
            return state_dict, metadata

    elif mode == "state_load" or mode == "quant_state_load":
        # Load the state_dict from the specified path
        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"State dictionary file not found at {state_dict_path}. Please ensure the model is saved correctly or the path and version are correct.")
        else:
            logger.info(f"Loading state_dict from {state_dict_path}")
            map_location = device if mode != "quant_state_load" else "cpu"  # Use CPU for quantized state load
            state_dict = torch.load(state_dict_path, map_location=map_location, weights_only=True)
            model: torch.nn.Module = init_mobilenetv2_cifar10_model(
                weights=None,  # No pretrained weights, we're loading our custom model
                to_quantize=(mode == "quant_state_load"),  # If mode is 'quant_state_load', we want a quantized model architecture
                is_quantized=(mode == "quant_state_load"),  # If mode is 'quant_state_load', we want a quantized model architecture
                num_classes=10  # CIFAR-10 has 10 classes
            )
            model.load_state_dict(state_dict)
            logger.info(f"Successfully loaded and applied state_dict from {state_dict_path}")
        
    elif mode == "jit_script":
        # Load the JIT scripted model
        if not os.path.exists(jit_script_path):
            raise FileNotFoundError(f"JIT scripted model file not found at {jit_script_path}. Please ensure the model is saved correctly or the path and version are correct.")
        else:
            logger.info(f"Loading JIT scripted model from {jit_script_path}")
            model: torch.jit.ScriptModule = torch.jit.load(jit_script_path, map_location=device)
            logger.info(f"Successfully loaded JIT scripted model from {jit_script_path}")
        
    elif mode == "jit_trace":
        # Load the JIT traced model
        if not os.path.exists(jit_trace_path):
            raise FileNotFoundError(f"JIT traced model file not found at {jit_trace_path}. Please ensure the model is saved correctly or the path and version are correct.")
        else:
            logger.info(f"Loading JIT traced model from {jit_trace_path}")
            model: torch.jit.ScriptModule = torch.jit.load(jit_trace_path, map_location=device)
            logger.info(f"Successfully loaded JIT traced model from {jit_trace_path}")

    # If the model is not quantized, we need to convert it to a quantized model if requested
    if convert_to_quantized and not mode == "quant_state_load":
        logger.info("Converting model to quantized version...")
        model = convert_mobilenetv2_cifar10_to_quantized(model)
        logger.info("Model converted to quantized version.")

    # If unstructured sparse config is present, apply it to reparametrize the model for further finetuning without touching to the pruned weights. This works because as pruned weights are already set to 0, pruning again with the same config will only prune the already pruned model weights.
    if "unstructured_sparse_config" in metadata and restore_unstruct_prune_reparam:
        unstruct_sparse_config = metadata["unstructured_sparse_config"]
        if unstruct_sparse_config:
            logger.info(f"Applying unstructured sparse config: {unstruct_sparse_config}")
            model = l1_unstructured_pruning(model, **unstruct_sparse_config)

    return model, metadata


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