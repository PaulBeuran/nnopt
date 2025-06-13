from typing import Callable, Literal
from itertools import islice

import torch
from torch.amp import autocast

import time
import onnxruntime as ort
from openvino.runtime import Core

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
                      f"Avg Batch Time: {avg_time_per_batch:.2f} ms | "  # Already in ms from _run_one_pass
                      f"Avg Sample Time: {avg_time_per_sample:.2f} ms") # Already in ms from _run_one_pass
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


def eval_onnx_model(
    onnx_model_path: str,
    test_dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.CrossEntropyLoss(),
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu",
    num_warmup_batches: int = 5,
    num_workers: int = 4,
    pin_memory: bool = True
) -> dict[str, float]:
    """
    Evaluate an ONNX model on a test dataset and return relevant metrics.
    Args:
        onnx_model_path (str): Path to the ONNX model file.
        test_dataset (torch.utils.data.Dataset): The dataset to evaluate the model on.
        batch_size (int): Batch size for evaluation.
        criterion (Callable): Loss function to use for evaluation.
        device (str): Device for PyTorch tensors ("cpu" or "cuda"). ONNX Runtime providers are set accordingly.
        num_warmup_batches (int): Number of batches to use for warmup phase.
        num_workers (int): Number of workers for DataLoader.
        pin_memory (bool): Whether to pin memory in DataLoader.
    Returns:
        dict[str, float]: Dictionary containing accuracy, avg_loss, samples_per_second, etc.
    """
    logger.info(f"Starting ONNX model evaluation for: {onnx_model_path}")
    logger.info(f"Evaluation on PyTorch device: {device}, batch size: {batch_size}")

    # Setup ONNX Runtime session
    providers = []
    if device == "cuda" and ort.get_device() == 'GPU':
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider') # Always include CPU as fallback or for CPU-only

    logger.info(f"Using ONNX Runtime providers: {providers}")
    try:
        ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
    except Exception as e:
        logger.error(f"Failed to create ONNX Runtime session: {e}")
        # Fallback to CPU if CUDA provider fails
        if 'CUDAExecutionProvider' in providers and len(providers) > 1:
            logger.warning("CUDAExecutionProvider failed. Falling back to CPUExecutionProvider.")
            providers = ['CPUExecutionProvider']
            ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
            logger.info(f"Using ONNX Runtime providers: {providers}")
        else:
            raise e


    input_name = ort_session.get_inputs()[0].name
    # Assuming the first output is the relevant one (e.g., logits)
    output_name = ort_session.get_outputs()[0].name 
    logger.info(f"ONNX Model Input Name: {input_name}, Output Name: {output_name}")

    # Create DataLoader for the test dataset
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Warmup phase
    if num_warmup_batches > 0 and len(test_loader) > num_warmup_batches:
        logger.info(f"Starting warmup for {num_warmup_batches} batches...")
        warmup_loader = islice(test_loader, num_warmup_batches)
        for inputs, _ in tqdm(warmup_loader, total=num_warmup_batches, desc="[ONNX Warmup]"):
            numpy_inputs = inputs.cpu().numpy() # ONNX Runtime typically expects CPU inputs
            ort_inputs = {input_name: numpy_inputs}
            _ = ort_session.run([output_name], ort_inputs)
        logger.info("Warmup complete.")

    # Main evaluation pass
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    total_inference_time = 0.0
    total_processed_samples_for_throughput = 0

    logger.info("Starting ONNX model evaluation pass...")
    for inputs, labels in tqdm(test_loader, desc="[ONNX Evaluation]"):
        batch_start_time = time.perf_counter()

        numpy_inputs = inputs.cpu().numpy() # Ensure input is on CPU for ONNX
        ort_inputs = {input_name: numpy_inputs}
        
        # Run inference
        ort_outputs = ort_session.run([output_name], ort_inputs)
        preds_numpy = ort_outputs[0] # Output is a list of numpy arrays

        batch_inference_time = (time.perf_counter() - batch_start_time) * 1000.0 # Convert to ms
        total_inference_time += batch_inference_time # Accumulates ms
        total_processed_samples_for_throughput += inputs.size(0)

        # Convert predictions to PyTorch tensor for metrics calculation
        # Output from ONNX is on CPU, move to target device for criterion
        preds_tensor = torch.from_numpy(preds_numpy).to(device)
        labels = labels.to(device)

        # Calculate loss
        loss = criterion(preds_tensor, labels)
        total_loss += loss.item() * inputs.size(0)

        # Calculate accuracy
        _, predicted_classes = torch.max(preds_tensor, 1)
        correct_predictions += (predicted_classes == labels).sum().item()
        total_samples += inputs.size(0)

    # Calculate final metrics
    avg_eval_loss = total_loss / total_samples if total_samples > 0 else 0
    eval_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    # total_inference_time is in ms
    samples_per_second = (total_processed_samples_for_throughput * 1000.0) / total_inference_time if total_inference_time > 0 else 0.0
    avg_time_per_batch = total_inference_time / len(test_loader) if len(test_loader) > 0 else 0.0 # in ms
    avg_time_per_sample = total_inference_time / total_processed_samples_for_throughput if total_processed_samples_for_throughput > 0 else 0.0 # in ms
    
    # Log throughput and system stats
    throughput_msg = (f"Throughput: {samples_per_second:.2f} samples/sec | "
                      f"Avg Batch Time: {avg_time_per_batch:.2f} ms | " # Already in ms
                      f"Avg Sample Time: {avg_time_per_sample:.2f} ms") # Already in ms
    stats_msg = _get_system_stats_msg(device) # System stats for the PyTorch part

    tqdm.write(f"ONNX Evaluation Complete: Avg Loss: {avg_eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")
    tqdm.write(throughput_msg)
    tqdm.write(f"System Stats (PyTorch side): {stats_msg}")

    return {
        "accuracy": eval_accuracy,
        "avg_loss": avg_eval_loss,
        "samples_per_second": samples_per_second,
        "avg_time_per_batch": avg_time_per_batch,
        "avg_time_per_sample": avg_time_per_sample,
        # Note: Parameter count like in PyTorch models is not directly applicable/easy for ONNX.
        # File size of the ONNX model could be a proxy for model size.
    }


def eval_model_openvino(
    onnx_model_path: str,
    test_dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.CrossEntropyLoss(),
    sparse_rate: float = 0.5,
    num_warmup_batches: int = 5,
    num_workers: int = 4,
    pin_memory: bool = True
) -> dict[str, float]:
    """
    Evaluate an ONNX model via OpenVINO with sparse‐weight decompression enabled.
    Returns accuracy, avg_loss, throughput & timing stats.
    """
    # 1. Compile ONNX model with sparse acceleration
    ie = Core()
    ov_model = ie.read_model(onnx_model_path)
    compiled = ie.compile_model(
        model=ov_model,
        device_name="CPU",
        config={"CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE": str(sparse_rate)}
    )

    # 2. Create DataLoader
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # 3. Warmup
    if num_warmup_batches > 0 and len(loader) > num_warmup_batches:
        for inputs, _ in islice(loader, num_warmup_batches):
            np_in = inputs.cpu().numpy()
            _ = compiled([np_in])

    # 4. Evaluation pass
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    total_inference_time = 0.0
    total_processed = 0

    tqdm.write(f"Starting OpenVINO evaluation for: {onnx_model_path}")
    for inputs, labels in tqdm(loader, desc="[OpenVINO Evaluation]"):
        bs = inputs.size(0)
        np_in = inputs.cpu().numpy()

        start = time.perf_counter()
        out = compiled([np_in])[0]               # (batch, num_classes)
        elapsed = (time.perf_counter() - start) * 1000.0 # Convert to ms

        # bookkeeping
        total_inference_time += elapsed # Accumulates ms
        total_processed += bs

        preds = torch.from_numpy(out)            # CPU tensor
        loss = criterion(preds, labels)
        total_loss += loss.item() * bs
        correct_predictions += (preds.argmax(dim=1) == labels).sum().item()
        total_samples += bs

    # Compute metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    # total_inference_time is in ms
    samples_per_second = (total_processed * 1000.0) / total_inference_time if total_inference_time > 0 else 0.0
    avg_time_per_batch = total_inference_time / len(loader) if len(loader) > 0 else 0.0 # in ms
    avg_time_per_sample = total_inference_time / total_processed if total_processed > 0 else 0.0 # in ms

    throughput_msg = (
        f"Throughput: {samples_per_second:.2f} samples/sec | "
        f"Avg Batch Time: {avg_time_per_batch:.2f} ms | " # Already in ms
        f"Avg Sample Time: {avg_time_per_sample:.2f} ms"  # Already in ms
    )
    stats_msg = _get_system_stats_msg("cpu")

    tqdm.write(f"OpenVINO Evaluation Complete: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    tqdm.write(throughput_msg)
    tqdm.write(f"System Stats: {stats_msg}")

    return {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "samples_per_second": samples_per_second,
        "avg_time_per_batch": avg_time_per_batch,
        "avg_time_per_sample": avg_time_per_sample,
    }