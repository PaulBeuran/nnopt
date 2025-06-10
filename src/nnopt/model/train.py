from typing import Callable, Literal

import torch
from torch.optim import Optimizer
from torch.amp import GradScaler

import logging
from tqdm import tqdm

from nnopt.model.utils import _run_one_pass, _get_system_stats_msg


# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)


def train_loop(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: Optimizer,
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu",
    use_amp: bool = True, 
    dtype: torch.dtype = torch.bfloat16 
) -> torch.nn.Module:
    """
    Train the model for a specified number of epochs, evaluating on a validation set after each epoch.
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        epochs (int): Number of epochs to train the model.
        criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Loss function.
        optimizer (Optimizer): Optimizer for training.
        device (Literal["cpu", "cuda"]): Device to run the model on.
        use_amp (bool): Whether to use automatic mixed precision.
        dtype (torch.dtype): Data type for mixed precision.
    """
    # Move model to device
    model.to(device)
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler(enabled=(use_amp and device == "cuda")) 

    # Training loop
    for epoch in range(epochs):
        # Train step
        train_results = _run_one_pass(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            dtype=dtype,
            is_train_pass=True,
            optimizer=optimizer,
            scaler=scaler,
            desc=f"Epoch {epoch+1}/{epochs} [Training]",
            enable_timing=True
        )
        
        # Validation step
        val_results = _run_one_pass(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            dtype=dtype,
            is_train_pass=False,
            desc=f"Epoch {epoch+1}/{epochs} [Validation]",
            enable_timing=True
        )
        
        # Log results
        stats_msg = _get_system_stats_msg(device)
        train_samples_per_second = train_results.get("samples_per_second", 0.0)
        train_accuracy = train_results.get("accuracy", 0.0)
        val_samples_per_second = val_results.get("samples_per_second", 0.0)

        tqdm.write(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {train_results['avg_loss']:.4f}, Train Acc: {train_accuracy:.4f}, Train Throughput: {train_samples_per_second:.2f} samples/s | "
            f"Val Loss: {val_results['avg_loss']:.4f}, Val Acc: {val_results['accuracy']:.4f}, Val Throughput: {val_samples_per_second:.2f} samples/s | "
            f"{stats_msg}"
        )

    return model


def train_model(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    epochs: int = 10,
    optimizer_cls: type[Optimizer] = torch.optim.Adam,  # Changed to optimizer_cls
    learning_rate: float = 0.001,
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu",
    use_amp: bool = True,
    dtype: torch.dtype = torch.bfloat16
) -> torch.nn.Module:
    """
    Train a model on a given dataset with specified parameters.
    
    Args:
        model (torch.nn.Module): The model to train.
        train_dataset (torch.utils.data.Dataset): Training dataset.
        val_dataset (torch.utils.data.Dataset): Validation dataset.
        batch_size (int): Batch size for training and validation.
        shuffle (bool): Whether to shuffle the training data.
        num_workers (int): Number of workers for DataLoader.
        pin_memory (bool): Whether to pin memory in DataLoader.
        epochs (int): Number of training epochs.
        optimizer_cls (type[Optimizer]): Optimizer class to use for training.
        learning_rate (float): Learning rate for the optimizer.
        device (Literal["cpu", "cuda"]): Device to run the model on.
        use_amp (bool): Whether to use automatic mixed precision.
        dtype (torch.dtype): Data type for mixed precision.

    Returns:
        torch.nn.Module: The trained model.
    """
    
    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=shuffle, 
                                               num_workers=num_workers, 
                                               pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size, 
                                             shuffle=False, 
                                             num_workers=num_workers, 
                                             pin_memory=pin_memory)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Move model to device
    model.to(device)

    # Optimizer
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)

    # Training loop
    train_loop(model, train_loader, val_loader, epochs, criterion, optimizer, device, use_amp, dtype)

    return model


def adapt_model_head_to_dataset(
    model: torch.nn.Module, 
    num_classes: int, 
    train_dataset: torch.utils.data.Dataset, 
    val_dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    head_train_epochs: int = 10,
    fine_tune_epochs: int = 5,
    optimizer_cls: type[Optimizer] = torch.optim.Adam,
    head_train_lr: float = 0.001,
    fine_tune_lr: float = 0.0001,
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu",
    use_amp: bool = True,
    dtype: torch.dtype = torch.bfloat16
) -> torch.nn.Module:
    """
    Adapt a pretrained classification model to a new dataset by replacing the final layer, training the head with the backbone frozen (this results in logistic regression training with the backbone acting as a feature extractor, to avoid catastrophic forgetting), and then fine-tuning the entire model on the new dataset (to adapt the backbone and the head jointly on the new dataset).
    Args:
        model (torch.nn.Module): The pretrained model to adapt.
        num_classes (int): Number of classes in the new dataset.
        train_dataset (torch.utils.data.Dataset): Training dataset.
        val_dataset (torch.utils.data.Dataset): Validation dataset.
        batch_size (int): Batch size for training and validation.
        shuffle (bool): Whether to shuffle the training data.
        num_workers (int): Number of workers for DataLoader.
        pin_memory (bool): Whether to pin memory in DataLoader.
        head_train_epochs (int): Number of epochs to train the head.
        fine_tune_epochs (int): Number of epochs to fine-tune the entire model.
        optimizer_cls (type[Optimizer]): Optimizer class to use for training.
        head_train_lr (float): Learning rate for training the head.
        fine_tune_lr (float): Learning rate for fine-tuning the entire model.
        device (Literal["cpu", "cuda"]): Device to run the model on.
        use_amp (bool): Whether to use automatic mixed precision.
        dtype (torch.dtype): Data type for mixed precision.
    """

    # Replace the final layer
    if hasattr(model, "fc"):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Sequential):
        # Common case for models like MobileNetV2, EfficientNet
        if hasattr(model.classifier[-1], "in_features"):
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
        else:
            # Fallback for other structures if needed, this might require specific handling
            logger.warning(f"Warning: Classifier final layer replacement might be inexact.")
            # Attempt to find the last linear layer if possible, or raise error
            for i in range(len(model.classifier) -1, -1, -1):
                if isinstance(model.classifier[i], torch.nn.Linear):
                    model.classifier[i] = torch.nn.Linear(model.classifier[i].in_features, num_classes)
                    break
            else:
                raise AttributeError(f"Could not find a final Linear layer in classifier")

    elif hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Linear): # e.g. some ViT models
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise AttributeError(f"""Model does not have "fc" or a known "classifier" structure to replace.""")


    model.to(device)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=shuffle, 
                                               num_workers=num_workers, 
                                               pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size, 
                                             shuffle=False, 
                                             num_workers=num_workers, 
                                             pin_memory=pin_memory)
    
    criterion = torch.nn.CrossEntropyLoss()

    # --- Train the head ---
    logger.info("Training head of the model with backbone frozen...")
    # Freeze the backbone
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final layer (fc or classifier)
    if hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, "classifier"):
        for param in model.classifier.parameters(): # Unfreeze all params in the classifier block
            param.requires_grad = True
    else: # Should not happen due to checks above
        raise AttributeError("Final layer not found for unfreezing.")

    optimizer_head = optimizer_cls(filter(lambda p: p.requires_grad, model.parameters()), lr=head_train_lr)
    
    train_loop(model, train_loader, val_loader, head_train_epochs, criterion, optimizer_head, device, use_amp, dtype)

    # --- Fine-tune the entire model ---
    logger.info("Fine-tuning full model...")
    # Unfreeze the backbone
    for param in model.parameters():
        param.requires_grad = True

    optimizer_finetune = optimizer_cls(model.parameters(), lr=fine_tune_lr)

    train_loop(model, train_loader, val_loader, fine_tune_epochs, criterion, optimizer_finetune, device, use_amp, dtype)

    return model