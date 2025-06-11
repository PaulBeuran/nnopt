from typing import Callable, Literal
from itertools import islice

import torch
from torch.amp import autocast

import logging
from tqdm import tqdm
from nnopt.model.utils import _run_one_pass, _get_system_stats_msg, count_parameters


# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)


# Function to evaluate a model on a test dataset and return the mean accuracy
def eval_model(
    model: torch.nn.Module,
    test_dataset: torch.utils.data.Dataset, # test_dataset is used for total_samples_for_throughput if loader is partial
    batch_size: int = 32,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.CrossEntropyLoss(),
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu",
    use_amp: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    num_warmup_batches: int = 5,
    num_workers: int = 4, 
    pin_memory: bool = True 
) -> dict[str, float]:
    """
    Evaluate a model on a test dataset and return the mean accuracy.
    Args:
        model (torch.nn.Module): The model to evaluate.
        test_dataset (torch.utils.data.Dataset): The dataset to evaluate the model on.
        batch_size (int): Batch size for evaluation.
        criterion (Callable): Loss function to use for evaluation.
        device (str): Device to run the evaluation on ("cpu" or "cuda").
        use_amp (bool): Whether to use automatic mixed precision.
        dtype (torch.dtype): Data type for mixed precision.
        num_warmup_batches (int): Number of batches to use for warmup phase.
        num_workers (int): Number of workers for DataLoader.
        pin_memory (bool): Whether to pin memory in DataLoader.
    Returns:
        float: Mean accuracy of the model on the test dataset.
    """
    logger.info(f"Starting evaluation on device: {device}, dtype: {dtype}, batch size: {batch_size}")
    # Ensure model is in evaluation mode
    model.to(device)
    # Set model to evaluation mode
    model.eval() 
    
    # Create DataLoader for the test dataset
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    # Warmup phase
    if num_warmup_batches > 0 and len(test_loader) > num_warmup_batches : 
        logger.info(f"Starting warmup for {num_warmup_batches} batches...")
        warmup_loader = islice(test_loader, num_warmup_batches)
        with torch.no_grad():
            for inputs, labels in tqdm(warmup_loader, total=num_warmup_batches, desc="[Warmup]"):
                inputs = inputs.to(device)
                with autocast(device_type=device, enabled=(use_amp and device == "cuda"), dtype=dtype if device == "cuda" else None):
                    _ = model(inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        logger.info("Warmup complete.")

    # Main evaluation pass
    eval_results = _run_one_pass(
        model=model,
        loader=test_loader, 
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        dtype=dtype,
        is_train_pass=False,
        desc="[Evaluation]",
        enable_timing=True # Crucial: ensure timing is enabled
    )

    # Extract evaluation metrics
    avg_eval_loss = eval_results["avg_loss"]
    eval_accuracy = eval_results["accuracy"]
    
    # Retrieve throughput metrics directly from eval_results
    samples_per_second = eval_results.get("samples_per_second", 0)
    avg_time_per_batch = eval_results.get("avg_time_per_batch", 0)
    avg_time_per_sample = eval_results.get("avg_time_per_sample", 0)

    # Log throughput and system stats
    throughput_msg = (f"Throughput: {samples_per_second:.2f} samples/sec | "
                      f"Avg Batch Time: {avg_time_per_batch*1000:.2f} ms | "
                      f"Avg Sample Time: {avg_time_per_sample*1000:.2f} ms")
    stats_msg = _get_system_stats_msg(device)
    
    tqdm.write(f"Evaluation Complete: Avg Loss: {avg_eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")
    tqdm.write(throughput_msg)
    tqdm.write(f"System Stats: {stats_msg}")
    
    return {
        "accuracy": eval_accuracy,
        "avg_loss": avg_eval_loss,
        "samples_per_second": samples_per_second,
        "avg_time_per_batch": avg_time_per_batch,
        "avg_time_per_sample": avg_time_per_sample,
        "params_stats": count_parameters(model),
    }