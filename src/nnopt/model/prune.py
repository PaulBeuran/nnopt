import torch
import torch.nn.utils.prune as unstruct_prune
import torch_pruning as struct_prune

import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)

# Unstructured pruning
def mark_weights_with_l1_unstructured_pruning(model: torch.nn.Module, 
                                  pruning_amount: float, 
                                  layers_to_prune: tuple = (torch.nn.Linear, torch.nn.Conv2d),
                                  parameter_name: str = "weight") -> torch.nn.Module:
    """
    Applies L1 unstructured pruning to specified layers of a model.

    Args:
        model (torch.nn.Module): The model to prune.
        pruning_amount (float): The fraction of connections to prune (e.g., 0.2 for 20%).
        layers_to_prune (tuple): A tuple of layer types to prune (e.g., (torch.nn.Linear, torch.nn.Conv2d)).
        parameter_name (str): The name of the parameter to prune within the layers (e.g., "weight", "bias").

    Returns:
        torch.nn.Module: The model with pruning applied (reparameterized).
    """
    if not (0.0 < pruning_amount < 1.0):
        raise ValueError("Pruning amount must be between 0.0 and 1.0 (exclusive).")

    logger.info(f"Applying L1 unstructured pruning with amount: {pruning_amount:.2f} for parameter '{parameter_name}' in layers: {[layer.__name__ for layer in layers_to_prune]}")
    num_pruned_layers = 0
    for module in model.modules():
        if isinstance(module, layers_to_prune):
            try:
                unstruct_prune.l1_unstructured(module, name=parameter_name, amount=pruning_amount)
                logger.debug(f"Pruned {parameter_name} of layer: {module}")
                num_pruned_layers +=1
            except Exception as e:
                logger.warning(f"Could not prune {parameter_name} of layer {module}: {e}")
    
    if num_pruned_layers == 0:
        logger.warning("No layers were pruned. Check 'layers_to_prune' and model structure.")
    else:
        logger.info(f"Applied L1 unstructured pruning to {num_pruned_layers} layers.")
    
    return model


def remove_pruning_reparameterization(model: torch.nn.Module,
                                      layers_to_prune: tuple = (torch.nn.Linear, torch.nn.Conv2d),
                                      parameter_name: str = "weight") -> torch.nn.Module:
    """
    Removes the pruning reparameterization, making the pruning permanent.
    The pruned weights are set to zero directly in the parameter tensor.

    Args:
        model (nn.Module): The model with pruning applied.
        layers_to_prune (tuple): A tuple of layer types from which to remove reparameterization.
        parameter_name (str): The name of the parameter that was pruned.

    Returns:
        nn.Module: The model with pruning made permanent.

    Notes:
        Must be called after `mark_weights_with_l1_unstructured_pruning`.
    """
    logger.info("Making pruning permanent by removing reparameterization...")
    num_permanent_layers = 0
    for module in model.modules():
        if isinstance(module, layers_to_prune):
            if unstruct_prune.is_pruned(module): # Check if the module has pruning hooks
                try:
                    unstruct_prune.remove(module, name=parameter_name)
                    logger.debug(f"Made pruning permanent for {parameter_name} of layer: {module}")
                    num_permanent_layers +=1
                except Exception as e:
                    logger.warning(f"Could not make pruning permanent for {parameter_name} of layer {module}: {e}")
            
    if num_permanent_layers == 0:
        logger.warning("No pruning reparameterization was removed. Was the model pruned?")
    else:
        logger.info(f"Made pruning permanent for {num_permanent_layers} layers.")
    return model


def calculate_sparsity(model: torch.nn.Module, 
                       layers_to_check: tuple = (torch.nn.Linear, torch.nn.Conv2d),
                       parameter_name: str = "weight") -> dict[str, float]:
    """
    Calculates the sparsity of specified parameters in the model.

    Args:
        model (nn.Module): The model to check.
        layers_to_check (tuple): Layer types to inspect.
        parameter_name (str): Name of the parameter (e.g., "weight").

    Returns:
        Dict[str, float]: A dictionary containing overall sparsity and sparsity per layer.

    Notes:
        Must be applied after `mark_weights_with_l1_unstructured_pruning` and `remove_pruning_reparameterization`.
    """
    results = {}
    total_zeros = 0
    total_elements = 0

    for name, module in model.named_modules():
        if isinstance(module, layers_to_check):
            if hasattr(module, parameter_name):
                param = getattr(module, parameter_name)
                if param is not None:
                    # If pruning is applied but not made permanent, the original tensor might not be zero.
                    # We need to check the 'weight_orig' and 'weight_mask' if they exist.
                    # However, if remove_pruning_reparameterization has been called,
                    # the 'weight' tensor itself will contain zeros.
                    
                    # For simplicity after `remove`, we check the actual parameter.
                    # If `remove` hasn't been called, this will report sparsity of the
                    # underlying tensor before the mask is applied during forward pass.
                    # For true sparsity *during* forward pass before `remove`, one would
                    # need to access module.weight_mask and module.weight_orig.
                    
                    layer_zeros = torch.sum(param.data == 0).item()
                    layer_elements = param.data.numel()
                    total_zeros += layer_zeros
                    total_elements += layer_elements
                    if layer_elements > 0:
                        layer_sparsity = layer_zeros / layer_elements
                        results[f"{name}.{parameter_name}_sparsity"] = layer_sparsity
                        logger.debug(f"Sparsity of {name}.{parameter_name}: {layer_sparsity:.4f} ({layer_zeros}/{layer_elements})")
                    else:
                        results[f"{name}.{parameter_name}_sparsity"] = 0.0
                        logger.debug(f"Layer {name}.{parameter_name} has 0 elements.")


    overall_sparsity = total_zeros / total_elements if total_elements > 0 else 0.0
    results["overall_sparsity"] = overall_sparsity
    logger.info(f"Overall sparsity ({parameter_name}): {overall_sparsity:.4f} ({total_zeros}/{total_elements})")
    
    return results


def apply_l1_unstructured_pruning(model: torch.nn.Module, 
                                  pruning_amount: float, 
                                  layers_to_prune: tuple = (torch.nn.Linear, torch.nn.Conv2d),
                                  parameter_name: str = "weight") -> torch.nn.Module:
    """
    Applies L1 unstructured pruning to the model and returns the pruned model.

    Args:
        model (torch.nn.Module): The model to prune.
        pruning_amount (float): The fraction of connections to prune.
        layers_to_prune (tuple): Layers to apply pruning to.
        parameter_name (str): The name of the parameter to prune.

    Returns:
        torch.nn.Module: The pruned model.
    """
    model = mark_weights_with_l1_unstructured_pruning(model, pruning_amount, layers_to_prune, parameter_name)
    model = remove_pruning_reparameterization(model, layers_to_prune, parameter_name)
    logger.info("L1 unstructured pruning completed.")
    _ = calculate_sparsity(model, layers_to_prune, parameter_name)
    return model


# Structured pruning
def count_model_parameters(model: torch.nn.Module, only_trainable: bool = True) -> int:
    """Counts the total number of parameters in a model."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def apply_l1_structured_pruning(
    model: torch.nn.Module,
    example_inputs: torch.Tensor,
    pruning_amount: float,
    layers_to_prune: tuple[type, ...] = (torch.nn.Conv2d, torch.nn.Linear),
    ignored_layers: list[torch.nn.Module] | None = None,
    prune_output_channels: bool = True
) -> torch.nn.Module:
    """
    Applies L1 magnitude structured pruning to specified layers of a model
    by removing a fraction of channels/features from each targeted layer.

    By default, it prunes output channels of Conv2d layers and output features
    of Linear layers.

    Args:
        model (nn.Module): The model to prune.
        example_inputs (torch.Tensor): A batch of example inputs for dependency tracing.
                                       Should be on the same device as the model.
        pruning_amount (float): The fraction of channels/features to prune from each
                                targeted layer (e.g., 0.2 for 20%).
        layers_to_prune (tuple[type, ...]): Tuple of layer types to prune.
        ignored_layers (list[torch.nn.Module] | None): A list of specific layer
                                                    modules to ignore during pruning.
        prune_output_channels (bool): If True, prunes output channels/features.
                                      If False, attempts to prune input channels/features
                                      (Note: Input channel importance calculation here is simplified).

    Returns:
        torch.nn.Module: The model with structured pruning applied. The model is modified in-place.
    """
    if not (0.0 <= pruning_amount < 1.0):
        raise ValueError("Pruning amount must be between 0.0 (inclusive) and 1.0 (exclusive).")

    if pruning_amount == 0.0:
        logger.info("Pruning amount is 0.0. No structured pruning will be applied.")
        return model

    # torch-pruning usually expects the model in eval mode for graph construction
    original_mode_is_train = model.training
    model.eval()

    device = next(model.parameters()).device
    example_inputs = example_inputs.to(device)

    logger.info(f"Applying L1 structured pruning with amount: {pruning_amount:.2f}")
    initial_params = count_model_parameters(model)
    logger.info(f"Initial model parameters: {initial_params}")

    DG = struct_prune.DependencyGraph()
    DG.build_dependency(model, example_inputs=example_inputs)

    num_pruned_overall_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, layers_to_prune):
            if ignored_layers and module in ignored_layers:
                logger.debug(f"Skipping ignored layer: {name} ({type(module).__name__})")
                continue

            current_channels = 0
            pruning_fn = None
            dim_type = ""
            weights = module.weight.data

            if prune_output_channels:
                if isinstance(module, torch.nn.Conv2d):
                    current_channels = module.out_channels
                    pruning_fn = struct_prune.prune_conv_out_channels
                    dim_type = "output channels"
                    # L1 norm for each output filter: (C_out, C_in, K_h, K_w) -> sum over C_in, K_h, K_w
                    channel_importance = torch.norm(weights.flatten(1), p=1, dim=1)
                elif isinstance(module, torch.nn.Linear):
                    current_channels = module.out_features
                    pruning_fn = struct_prune.prune_linear_out_features
                    dim_type = "output features"
                    # L1 norm for each output feature's weights: (F_out, F_in) -> sum over F_in
                    channel_importance = torch.norm(weights, p=1, dim=1)
                else: # Should not be reached if layers_to_prune is respected
                    continue
            else: # Pruning input channels
                if isinstance(module, torch.nn.Conv2d):
                    current_channels = module.in_channels
                    pruning_fn = struct_prune.prune_conv_in_channels
                    dim_type = "input channels"
                    # Simplified L1 for input channels: (C_out, C_in, K_h, K_w) -> transpose to (C_in, C_out, K_h, K_w)
                    channel_importance = torch.norm(weights.transpose(0,1).contiguous().flatten(1), p=1, dim=1)
                elif isinstance(module, torch.nn.Linear):
                    current_channels = module.in_features
                    pruning_fn = struct_prune.prune_linear_in_features
                    dim_type = "input features"
                    # Simplified L1 for input features: (F_out, F_in) -> transpose to (F_in, F_out)
                    channel_importance = torch.norm(weights.T.contiguous(), p=1, dim=1)
                else: # Should not be reached
                    continue

            if current_channels == 0:
                logger.warning(f"Layer {name} ({type(module).__name__}) has 0 {dim_type} to prune. Skipping.")
                continue

            num_to_prune = int(pruning_amount * current_channels)

            if num_to_prune == 0:
                logger.debug(f"Layer {name}: No {dim_type} to prune with amount {pruning_amount:.2f} (Total: {current_channels}).")
                continue

            # Ensure we don't prune all channels, as torch-pruning might error or lead to a dead network.
            # It's safer to leave at least one channel.
            if num_to_prune >= current_channels:
                num_to_prune = current_channels - 1
                logger.warning(
                    f"Layer {name}: Pruning amount {pruning_amount:.2f} would remove all or too many {dim_type}. "
                    f"Adjusting to prune {num_to_prune} {dim_type} to keep at least one."
                )
                if num_to_prune <= 0: # If current_channels was 1
                    logger.info(f"Layer {name}: Cannot prune, only 1 {dim_type} exists. Skipping.")
                    continue
            
            # Get indices of channels to prune (those with smallest L1 norm)
            sorted_channel_indices = torch.argsort(channel_importance)
            pruning_indices = sorted_channel_indices[:num_to_prune].tolist()

            try:
                pruning_plan = DG.get_pruning_plan(module, pruning_fn, idxs=pruning_indices)
                if pruning_plan:
                    logger.debug(f"Pruning {num_to_prune} {dim_type} from layer {name} ({type(module).__name__}). Smallest L1 norm indices (first 10): {pruning_indices[:10]}...")
                    pruning_plan.exec()
                    num_pruned_overall_layers += 1
                else:
                    logger.warning(f"Could not generate pruning plan for layer {name} ({type(module).__name__}).")
            except Exception as e:
                logger.error(f"Failed to prune layer {name} ({type(module).__name__}): {e}", exc_info=True)

    if num_pruned_overall_layers == 0:
        logger.warning("No layers were structurally pruned. Check 'layers_to_prune', model structure, and pruning_amount.")
    else:
        logger.info(f"Applied structured pruning to {num_pruned_overall_layers} layers.")

    final_params = count_model_parameters(model)
    logger.info(f"Final model parameters after structured pruning: {final_params}")
    if initial_params > 0 :
        reduction_percent = (initial_params - final_params) / initial_params * 100
        logger.info(f"Parameter reduction: {initial_params - final_params} ({reduction_percent:.2f}%)")
    else:
        logger.info(f"Parameter reduction: {initial_params - final_params}")


    if original_mode_is_train:
        model.train() # Set back to original mode

    return model