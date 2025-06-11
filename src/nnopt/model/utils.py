from typing import Callable

import os
import time

import torch
import torch.ao.nn.quantized as nnq
from torch.optim import Optimizer
from torch.amp import GradScaler, autocast

import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)

import psutil
try:
    import pynvml
    pynvml.nvmlInit()
    pynvml_available = True
except (ImportError, pynvml.NVMLError):
    pynvml_available = False
logger.debug(f"pynvml available: {pynvml_available}")

def _get_system_stats_msg(device: str) -> str:
    """
    Get a formatted string with current system stats including CPU, RAM, and GPU usage.
    Args:
        device (str): The device type ("cpu" or "cuda").
    Returns:
        str: Formatted string with system stats.
    """
    cpu_usage = psutil.cpu_percent()
    
    vm = psutil.virtual_memory()
    ram_used_gb = vm.used / (1024**3)
    ram_total_gb = vm.total / (1024**3)
    ram_msg = f"{ram_used_gb:.1f}/{ram_total_gb:.1f}GB ({vm.percent:.1f}%)"
    
    stats_msg = f"CPU Usage: {cpu_usage:.2f}% | RAM Usage: {ram_msg}"

    if device == "cuda" and pynvml_available and torch.cuda.is_available():
        try:
            gpu_id = torch.cuda.current_device()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used_gb = mem_info.used / (1024**3)
            vram_total_gb = mem_info.total / (1024**3)
            gpu_mem_percent = mem_info.used / mem_info.total * 100
            vram_msg = f"{vram_used_gb:.1f}/{vram_total_gb:.1f}GB ({gpu_mem_percent:.1f}%)"
            
            stats_msg += f" | GPU {gpu_id} Util: {gpu_util:.2f}% | GPU {gpu_id} Mem: {vram_msg}"
        except pynvml.NVMLError as e:
            stats_msg += f" | GPU Stats Error: {e}"
    return stats_msg


def count_parameters(model: torch.nn.Module) -> dict[str, float]:
    """
    Counts parameters in a PyTorch model, distinguishing between
    INT8 weights, FP32 weights, and various FP32 bias/other parameters.
    Works for both standard FP32 models and quantized (INT8) models.
    Args:
        model (torch.nn.Module): The model to analyze.
    Returns:
        dict: A dictionary with counts of different parameter types:
            - "int_weight_params": Number of INT8 weight parameters
            - "float_weight_params": Number of FP32 weight parameters
            - "float_bias_params": Number of FP32 bias parameters
            - "bn_param_params": Number of FP32 BatchNorm parameters (weight/bias)
            - "other_float_params": Number of other FP32 parameters
            - "total_params": Total number of parameters (INT8 + FP32)
            - "approx_memory_mb_for_params": Approximate memory footprint in MB
    """
    int_weight_elements = 0
    float_weight_elements = 0  # For FP32 weights
    float_bias_elements = 0    # For biases (typically float)
    bn_param_elements = 0      # For BatchNorm weight/bias (typically float)
    other_float_params = 0     # For any other float parameters

    counted_modules = set()

    for name, module in model.named_modules():
        if module in counted_modules: # Avoid double counting if modules are shared or containers
            continue
        counted_modules.add(module)

        # Quantized Layers (nnq.*)
        if isinstance(module, (nnq.Linear, nnq.Conv2d, nnq.Conv1d, nnq.Conv3d)):
            # Weight is a quantized tensor
            q_weight = module.weight()
            int_weight_elements += q_weight.int_repr().numel()
            
            # Bias is typically float for these layers after default PTQ
            if module.bias() is not None:
                float_bias_elements += module.bias().numel()
        
        # Standard FP32 Layers (nn.*)
        elif isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)):
            if hasattr(module, 'weight') and module.weight is not None:
                float_weight_elements += module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                float_bias_elements += module.bias.numel()
        
        # BatchNorm Layers (can be nnq.BatchNorm*d or nn.BatchNorm*d)
        # Both nnq.BatchNorm*d and nn.BatchNorm*d have FP32 weight and bias Parameters
        elif isinstance(module, (nnq.BatchNorm2d, nnq.BatchNorm3d, 
                                 torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            if hasattr(module, 'weight') and module.weight is not None:
                 bn_param_elements += module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                 bn_param_elements += module.bias.numel()
            # running_mean and running_var are buffers, not parameters.
            
        # For other module types that might have parameters (e.g., nn.Embedding)
        # This ensures we count parameters of modules not explicitly listed above.
        else:
            # Check for parameters directly owned by this module, not its children,
            # as children will be covered by the loop over named_modules().
            for param_name, param in module.named_parameters(recurse=False):
                other_float_params += param.numel()
    
    total_int_elements = int_weight_elements
    total_float_elements = float_weight_elements + float_bias_elements + bn_param_elements + other_float_params
    total_elements = total_int_elements + total_float_elements
    
    # Memory footprint: INT8 = 1 byte, Float32 = 4 bytes
    memory_bytes = (total_int_elements * 1) + (total_float_elements * 4)
    
    return {
        "int_weight_params": int_weight_elements,
        "float_weight_params": float_weight_elements,
        "float_bias_params": float_bias_elements,
        "bn_param_params": bn_param_elements,
        "other_float_params": other_float_params,
        "total_params": total_elements,
        "approx_memory_mb_for_params": memory_bytes / (1024**2)
    }

# Helper function to run one pass (train, validation, or evaluation)
def _run_one_pass(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
    use_amp: bool,
    dtype: torch.dtype,
    is_train_pass: bool,
    optimizer: Optimizer | None = None,
    scaler: GradScaler | None = None,
    desc: str = "Processing",
    enable_timing: bool = False
) -> dict[str, float]:
    """
    Run one pass of the model (training or evaluation) on the provided data loader.
    Args:
        model (torch.nn.Module): The model to run.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Loss function.
        device (str): Device to run the model on ("cpu" or "cuda").
        use_amp (bool): Whether to use automatic mixed precision.
        dtype (torch.dtype): Data type for mixed precision.
        is_train_pass (bool): Whether this is a training pass.
        optimizer (Optimizer | None): Optimizer for training pass, if applicable.
        scaler (GradScaler | None): GradScaler for mixed precision, if applicable.
        desc (str): Description for progress bar.
        enable_timing (bool): Whether to enable timing for performance metrics.
    """
    if is_train_pass:
        model.train()
        if optimizer is None:
            raise ValueError("Optimizer must be provided for training pass.")
    else:
        model.eval()

    total_loss = 0.0
    correct_preds = 0
    total_samples_processed = 0
    
    batch_times = []
    
    gpu_handle = None
    if device == "cuda" and pynvml_available:
        try:
            gpu_id = torch.cuda.current_device()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        except pynvml.NVMLError as e:
            logger.warning(f"Could not get GPU handle: {e}")
            gpu_handle = None

    pbar = tqdm(loader, desc=desc)
    
    context_manager = torch.no_grad() if not is_train_pass else torch.enable_grad()
    with context_manager:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            batch_start_time = 0.0 
            if enable_timing:
                if device == "cuda": torch.cuda.synchronize()
                batch_start_time = time.perf_counter()

            if is_train_pass and optimizer:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device, enabled=(use_amp and device == "cuda"), dtype=dtype if device == "cuda" else None):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            if is_train_pass and optimizer: 
                if scaler and use_amp and device == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            
            current_batch_duration = 0.0
            if enable_timing:
                if device == "cuda": torch.cuda.synchronize()
                batch_end_time = time.perf_counter()
                current_batch_duration = batch_end_time - batch_start_time
                batch_times.append(current_batch_duration)

            total_loss += loss.item() * inputs.size(0) 
            _, predicted = outputs.max(1)
            correct_preds += predicted.eq(labels).sum().item()
            total_samples_processed += labels.size(0)

            # --- Live stats for tqdm postfix ---
            postfix_stats = {"loss": f"{loss.item():.4f}"}
            current_acc = correct_preds / total_samples_processed if total_samples_processed > 0 else 0
            postfix_stats["acc"] = f"{current_acc:.4f}"

            if enable_timing and current_batch_duration > 0:
                samples_this_batch = inputs.size(0)
                batch_throughput = samples_this_batch / current_batch_duration
                postfix_stats["samples/s"] = f"{batch_throughput:.1f}" # samples per second
            
            # System stats
            postfix_stats["cpu"] = f"{psutil.cpu_percent():.1f}%"
            
            vm = psutil.virtual_memory()
            ram_used_gb = vm.used / (1024**3)
            ram_total_gb = vm.total / (1024**3)
            postfix_stats["ram"] = f"{ram_used_gb:.1f}/{ram_total_gb:.1f}GB ({vm.percent:.1f}%)"

            if device == "cuda" and pynvml_available and gpu_handle:
                try:
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    vram_used_gb = mem_info.used / (1024**3)
                    vram_total_gb = mem_info.total / (1024**3)
                    gpu_mem_percent = mem_info.used / mem_info.total * 100
                    postfix_stats["gpu_util"] = f"{gpu_util:.1f}%"
                    postfix_stats["gpu_mem"] = f"{vram_used_gb:.1f}/{vram_total_gb:.1f}GB ({gpu_mem_percent:.1f}%)"
                except pynvml.NVMLError:
                    postfix_stats["gpu_util"] = "N/A"
                    postfix_stats["gpu_mem"] = "N/A"
            
            pbar.set_postfix(**postfix_stats)
            # --- End live stats ---

    avg_loss = total_loss / total_samples_processed if total_samples_processed > 0 else 0
    accuracy = correct_preds / total_samples_processed if total_samples_processed > 0 else 0
    
    results = {"avg_loss": avg_loss, "accuracy": accuracy, "total_samples_processed": float(total_samples_processed)}
    
    if enable_timing and batch_times: 
        total_inference_time = sum(batch_times)
        num_timed_batches = len(batch_times)
        results["total_inference_time"] = total_inference_time
        results["num_timed_batches"] = float(num_timed_batches)
        
        results["avg_time_per_batch"] = total_inference_time / num_timed_batches if num_timed_batches > 0 else 0
        results["avg_time_per_sample"] = total_inference_time / total_samples_processed if total_samples_processed > 0 else 0
        results["samples_per_second"] = total_samples_processed / total_inference_time if total_inference_time > 0 else 0
    else: 
        results["total_inference_time"] = 0.0
        results["num_timed_batches"] = 0.0
        results["avg_time_per_batch"] = 0.0
        results["avg_time_per_sample"] = 0.0
        results["samples_per_second"] = 0.0

    return results
