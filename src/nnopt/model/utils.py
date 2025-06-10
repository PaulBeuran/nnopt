from typing import Callable

import os
import time

import torch
from torch.optim import Optimizer
from torch.amp import GradScaler, autocast

import logging
from tqdm import tqdm

import psutil
try:
    import pynvml
    pynvml.nvmlInit()
    pynvml_available = True
except (ImportError, pynvml.NVMLError):
    pynvml_available = False
print(f"pynvml available: {pynvml_available}")

# Setup logging
logging.basicConfig(level=logging.DEBUG, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)


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
