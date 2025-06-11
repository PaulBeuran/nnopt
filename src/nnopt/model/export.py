\
import torch
import os
import logging

# Try to import TensorRT
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    # print("TensorRT Python bindings are not installed. TensorRT engine creation will be skipped or done via trtexec.")

# Setup logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler()])

def export_model_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    onnx_path: str,
    input_names: list[str] = ['input'],
    output_names: list[str] = ['output'],
    dynamic_axes: dict | None = None,
    opset_version: int = 13
) -> bool:
    """
    Exports a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        dummy_input (torch.Tensor): A dummy input tensor for tracing the model.
                                    The shape and dtype should match the model's expected input.
        onnx_path (str): Path to save the ONNX model.
        input_names (list[str]): Names for the input nodes in the ONNX graph.
        output_names (list[str]): Names for the output nodes in the ONNX graph.
        dynamic_axes (dict | None): A dictionary specifying dynamic axes for inputs/outputs.
                                    Example: {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        opset_version (int): The ONNX opset version to use.

    Returns:
        bool: True if export was successful, False otherwise.
    """
    logger.info(f"Starting ONNX export to {onnx_path} with opset_version={opset_version}...")
    model.eval()  # Ensure the model is in evaluation mode

    # Ensure the directory for the ONNX file exists
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            export_params=True,
            do_constant_folding=True,
        )
        logger.info(f"Model successfully exported to {onnx_path}")
        return True
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return False

def convert_onnx_to_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    use_fp16: bool = False,
    use_int8: bool = False,
    max_batch_size: int = 1, # This is used if not using explicit batch with optimization profiles
    workspace_size_gb: int = 1,
    optimization_profiles: list[tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]] | None = None
) -> bool:
    """
    Converts an ONNX model to a TensorRT engine.

    Args:
        onnx_path (str): Path to the ONNX model file.
        engine_path (str): Path to save the TensorRT engine.
        use_fp16 (bool): Enable FP16 mode if supported.
        use_int8 (bool): Enable INT8 mode if supported (requires calibration or QAT model).
        max_batch_size (int): Maximum batch size for the engine (used with implicit batch).
        workspace_size_gb (int): Maximum workspace size in GB for TensorRT.
        optimization_profiles (list[tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]] | None):
            A list of optimization profiles for dynamic input shapes. Each tuple should contain:
            (input_name, min_shape, opt_shape, max_shape).
            Example: [("input", (1, 3, 224, 224), (8, 3, 224, 224), (16, 3, 224, 224))]
            If None, assumes static shapes or uses implicit batch with max_batch_size.

    Returns:
        bool: True if engine creation was successful, False otherwise.
    """
    if not TRT_AVAILABLE:
        logger.error("TensorRT Python bindings (tensorrt) are not available. Cannot create TensorRT engine.")
        # As a fallback, you could suggest using trtexec here if desired.
        # For example:
        # cmd = f"trtexec --onnx={onnx_path} --saveEngine={engine_path}"
        # if use_fp16: cmd += " --fp16"
        # if use_int8: cmd += " --int8"
        # logger.info(f"To build the engine manually, you can try: \\n{cmd}")
        return False

    if not os.path.exists(onnx_path):
        logger.error(f"ONNX file not found: {onnx_path}")
        return False

    logger.info(f"Starting TensorRT engine creation from {onnx_path} to {engine_path}...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # Or trt.Logger.INFO for more verbosity

    # Ensure the directory for the engine file exists
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    # 1. Create a builder
    builder = trt.Builder(TRT_LOGGER)

    # 2. Create a network definition
    # Explicit batch is recommended for new models/TensorRT versions
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # 3. Create an ONNX parser
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 4. Parse the ONNX model
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            logger.error("Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            return False
    logger.info("ONNX model parsed successfully.")

    # 5. Create a builder configuration
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_gb * (1024 ** 3))


    # Set precision flags
    if use_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode enabled.")
        else:
            logger.warning("FP16 mode requested but not supported or not fast on this platform.")
    
    if use_int8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # If using PTQ ONNX models, you might need a calibrator here unless the ONNX has Q/DQ nodes.
            # For QAT models, the Q/DQ nodes in ONNX should be sufficient.
            logger.info("INT8 mode enabled.")
        else:
            logger.warning("INT8 mode requested but not supported or not fast on this platform.")

    # Handle dynamic shapes with optimization profiles
    if optimization_profiles:
        for profile_data in optimization_profiles:
            input_name, min_shape, opt_shape, max_shape = profile_data
            profile = builder.create_optimization_profile()
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
        logger.info(f"{len(optimization_profiles)} optimization profile(s) added.")
    elif (network_flags >> int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) & 1:
        # If explicit batch but no profiles, it implies static shapes defined in ONNX
        logger.info("Using explicit batch with static shapes from ONNX (no optimization profiles provided).")
    else: # Implicit batch
        builder.max_batch_size = max_batch_size # Deprecated for explicit batch networks
        logger.info(f"Using implicit batch with max_batch_size = {max_batch_size}.")


    # 6. Build the engine
    logger.info("Building TensorRT engine... (This may take a while)")
    serialized_engine = None
    try:
        # It's good practice to check if the network has an implicit batch dimension if not using explicit batch
        if not ((network_flags >> int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) & 1) and network.has_implicit_batch_dimension:
             logger.info(f"Network has implicit batch dimension. Max batch size for builder: {builder.max_batch_size}")
        
        serialized_engine = builder.build_serialized_network(network, config)
    except Exception as e:
        logger.error(f"TensorRT engine build failed: {e}")
        # Fallback for older TensorRT versions that might not have build_serialized_network
        # or if there's an issue with it.
        logger.info("Trying with build_engine...")
        try:
            engine = builder.build_engine(network, config)
            if engine:
                serialized_engine = engine.serialize()
            else:
                logger.error("build_engine also failed to produce an engine.")
                return False
        except Exception as e2:
            logger.error(f"Fallback build_engine also failed: {e2}")
            return False


    if not serialized_engine:
        logger.error("Failed to build TensorRT engine (serialized_engine is None).")
        return False
    
    logger.info("TensorRT engine built successfully.")

    # 7. Save the engine
    try:
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        logger.info(f"TensorRT engine saved to {engine_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save TensorRT engine: {e}")
        return False