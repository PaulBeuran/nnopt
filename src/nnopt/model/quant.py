import torch
import torch.quantization
import logging # Ensure logging is imported

from nnopt.model.train import train_model
from nnopt.model.eval import eval_model
from nnopt.model.const import TORCH_BACKENDS_QUANTIZED_ENGINE, DEVICE, DTYPE
from nnopt.model.prune import remove_pruning_reparameterization
from tqdm import tqdm


# Setup backend for quantization
torch.backends.quantized.engine = TORCH_BACKENDS_QUANTIZED_ENGINE

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)


def post_training_quantization(
    model: torch.nn.Module, 
    val_dataset: torch.utils.data.Dataset,
    num_calibration_batches: int = 10,
    batch_size: int = 32
) -> torch.nn.Module:
    """
    Applies post-training quantization to the given model in-place.
    Pruning reparameterization is removed and model fusion is attempted before quantization.

    Args:
        model (torch.nn.Module): The FP32 model to quantize (will be modified in-place).
        val_dataset (torch.utils.data.Dataset): The validation dataset for calibration.
        num_calibration_batches (int): Number of batches to use for calibration.
        batch_size (int): Batch size for calibration.

    Returns:
        torch.nn.Module: The quantized model
    """
    logger.info("Starting in-place post-training quantization...")

    model.cpu()  # Ensure model is on CPU for quantization steps
    model.eval() # Set model to evaluation mode

    # Remove pruning reparameterization (operates in-place)
    logger.info("Attempting to remove pruning reparameterization...")
    remove_pruning_reparameterization(model) # Assumes this modifies model in-place
    logger.info("Pruning reparameterization removal step completed.")

    # Fuse modules if possible (typically in-place)
    if hasattr(model, 'fuse_model') and callable(model.fuse_model):
        logger.info("Attempting to fuse model modules...")
        model.fuse_model() # Standard torchvision models fuse in-place
        logger.info("Model fusion step completed.")
    else:
        logger.info("Model does not have a fuse_model method or it's not callable. Skipping fusion.")

    # Set qconfig on the model.
    model.qconfig = torch.quantization.get_default_qconfig(torch.backends.quantized.engine)
    
    # Prepare the model. Using inplace=True.
    logger.info("Preparing model for quantization (inserting observers)...")
    torch.quantization.prepare(model, inplace=True)
    model.eval() # Ensure it's still in eval mode after prepare
    logger.info("Model prepared for quantization.")

    # Calibrate using the prepared model.
    logger.info(f"Starting calibration with {num_calibration_batches} batches...")
    with torch.no_grad():
        model.to(DEVICE)
        data_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        actual_calib_batches = min(num_calibration_batches, len(data_loader))
        for i, (inputs, _) in tqdm(enumerate(data_loader), total=actual_calib_batches):
            if i >= actual_calib_batches:
                break
            model(inputs.to(DEVICE)) # Pass inputs to the model for calibration
    logger.info(f"Calibration completed over {actual_calib_batches} batches.")

    # Convert the prepared model. Using inplace=True.
    logger.info("Converting model to quantized version...")
    # Model is already on CPU and in eval mode
    model.cpu()  # Ensure model is on CPU for conversion
    torch.quantization.convert(model, inplace=True)
    model.eval() # Ensure it's still in eval mode after convert
    logger.info("Model quantization completed.")

    # Evaluate the fully quantized model.
    logger.info("Starting evaluation of the quantized model...")
    ptq_metrics = eval_model(
        model=model, # Evaluate the in-place modified model
        test_dataset=val_dataset,
        batch_size=batch_size,
        device="cpu",      
        use_amp=False,     
        dtype=torch.float32 
    )
    logger.info(f"Post-training quantization metrics: {ptq_metrics}")

    return model # Return the in-place modified model


def quantization_aware_training(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    epochs: int = 10,
    batch_size: int = 32,
    training_device: str = DEVICE
) -> torch.nn.Module:
    """
    Applies quantization-aware training to the given model.
    Args:
        model (torch.nn.Module): The FP32 model to train with QAT.
                                   If pruned, pruning reparameterization should be active.
        train_dataset (torch.utils.data.Dataset): Training dataset.
        val_dataset (torch.utils.data.Dataset): Validation dataset for evaluation during QAT.
        epochs (int): Number of epochs for QAT fine-tuning.
        batch_size (int): Batch size for QAT.
        training_device (str): Device for training ("cpu" or "cuda").
    Returns:
        torch.nn.Module: The QAT-trained model, converted to a fully quantized model.
    """
    logger.info("Preparing model for quantization-aware training...")
    
    model_to_qat = model
    model_to_qat.cpu()

    # Fuse modules if possible (typically in-place)
    if hasattr(model, 'fuse_model') and callable(model.fuse_model):
        logger.info("Attempting to fuse model modules...")
        model.fuse_model() # Standard torchvision models fuse in-place
        logger.info("Model fusion step completed.")
    else:
        logger.info("Model does not have a fuse_model method or it's not callable. Skipping fusion.")

    logger.info("Making pruning permanent (if model was pruned) before INT8 conversion...")
    # Using default layers and parameter name from remove_pruning_reparameterization,
    # assuming they match how pruning was initially applied.
    remove_pruning_reparameterization(model_to_qat) # Modifies model_to_qat in-place
    logger.info("Pruning reparameterization removed.")

    model_to_qat.train() 

    model_to_qat.qconfig = torch.quantization.get_default_qat_qconfig(torch.backends.quantized.engine)
    
    logger.info("Preparing model for QAT (inserting FakeQuantize modules)...")
    model_prepared_qat = torch.quantization.prepare_qat(model_to_qat, inplace=True)
    model_prepared_qat.train() 
    logger.info("Model prepared for quantization-aware training.")

    logger.info(f"Starting quantization-aware training for {epochs} epochs on {training_device}...")
    model_prepared_qat.to(training_device) 
    
    trained_model_qat = train_model( 
        model=model_prepared_qat,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        device=training_device, # Can be "cpu" or "cuda"
        dtype=torch.float32, # Must be float32 for QAT
        use_amp=False # AMP can't be used with QAT
    )

    logger.info("Converting QAT model to a fully quantized model (INT8)...")
    trained_model_qat.cpu()
    model_quantized = torch.quantization.convert(trained_model_qat, inplace=True)
    model_quantized.eval()
    logger.info("QAT model converted to INT8.")

    logger.info("Starting evaluation of the QAT quantized model...")
    qat_metrics = eval_model(
        model=model_quantized,
        test_dataset=val_dataset,
        batch_size=batch_size,
        device="cpu", 
        use_amp=False,
        dtype=torch.float32
    )
    logger.info(f"Quantization-aware training metrics: {qat_metrics}")

    return model_quantized