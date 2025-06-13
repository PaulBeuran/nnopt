import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional

# Define a type alias for model style for clarity
ModelStyle = Dict[str, str]

def plot_model_performance_scatter(
    model_names: List[str],
    test_accuracies: List[float],
    cpu_time_per_sample_test: List[float],
    total_params_mb_test: List[float],
    model_styles: Optional[List[Optional[Dict[str, str]]]] = None, # Allow None for individual styles,
    output_dir: str = None
) -> None:
    """
    Generates and saves scatter plots for model performance analysis.
    Each model point can have a specified color and marker.
    The legend maps styles to model names.

    Args:
        model_names: List of model names.
        test_accuracies: List of test accuracies for each model.
        cpu_time_per_sample_test: List of average CPU inference time per sample for each model.
        total_params_mb_test: List of model sizes in MB for each model.
        model_styles: Optional list of dictionaries (or None for default style for a model)
                      containing style attributes ('color', 'marker') for each model.
        output_dir: Directory to save the plots. If None, plots will be shown instead.
    """
    num_models = len(model_names)

    # Validate model_styles if provided
    if model_styles and len(model_styles) != num_models:
        raise ValueError("Length of model_styles must match length of model_names if provided.")

    # Generate default colors if no styles are provided or if a style is incomplete
    default_color_map = None
    if num_models <= 10:
        default_color_map = plt.cm.get_cmap('tab10', num_models)
    else:
        default_color_map = plt.cm.get_cmap('viridis', num_models)

    # Scatter Plot: Accuracy vs. Time per Sample
    fig_time, ax_time = plt.subplots(figsize=(12, 8))
    for i in range(num_models):
        current_style = model_styles[i] if model_styles else None
        
        color = default_color_map(i) # Default color
        marker = 'o'  # Default marker

        if current_style:
            color = current_style.get('color', color) # Use specified or default
            marker = current_style.get('marker', marker) # Use specified or default

        ax_time.scatter(cpu_time_per_sample_test[i], test_accuracies[i], marker=marker, s=100,
                        color=color, label=model_names[i], alpha=0.9)

    ax_time.set_xlabel('Average Time per Sample (ms) on CPU')
    ax_time.set_ylabel('Accuracy')
    ax_time.set_title('Accuracy vs. Inference Time per Sample on CPU (Test Metrics)')
    ax_time.legend(loc='best', title="Models")
    ax_time.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if output_dir:
        plt.savefig(f"{output_dir}/accuracy_vs_time.png")
        plt.close(fig_time)
    else:
        plt.show()

    # Scatter Plot: Accuracy vs. Memory Usage
    fig_mem, ax_mem = plt.subplots(figsize=(12, 8))
    for i in range(num_models):
        current_style = model_styles[i] if model_styles else None

        color = default_color_map(i) # Default color
        marker = 'o'  # Default marker

        if current_style:
            color = current_style.get('color', color)
            marker = current_style.get('marker', marker)

        ax_mem.scatter(total_params_mb_test[i], test_accuracies[i], marker=marker, s=100,
                       color=color, label=model_names[i], alpha=0.9)

    ax_mem.set_xlabel('Model Size (MB)')
    ax_mem.set_ylabel('Accuracy')
    ax_mem.set_title('Accuracy vs. Model Size (Test Metrics)')
    ax_mem.legend(loc='best', title="Models")
    ax_mem.grid(True, linestyle='--', alpha=0.7)
    # ax_mem.set_xscale('log') # Optional: if memory sizes vary a lot
    plt.tight_layout()
    if output_dir:
        plt.savefig(f"{output_dir}/accuracy_vs_memory.png")
        plt.close(fig_mem)
    else:
        plt.show()