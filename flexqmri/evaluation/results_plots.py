"""Plotting functions for comparing and visualising experiment results."""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless/SSH environments
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from flexqmri.evaluation.utils import get_modality_parameters, get_all_modality_params, shorten_labels

def plot_scatter_predictions(
    results: dict,
    modality: str,
    spearman_r: dict | None = None,
    save_path: str = None,
) -> None:
    """Plot scatter of predicted vs true values, one subplot per biophysical parameter.

    Args:
        results (dict): Results dict with 'predictions' and 'y_true' lists of tensors.
        modality (str): Modality of the data ('ivim', 't2star', …).
        spearman_r (dict, optional): Output of compute_spearman_r; if provided, annotates
            each subplot with the Spearman ρ between pred and true.
        save_path (str, optional): If provided, save the figure to this path.

    Returns:
        None
    """
    parameters = get_modality_parameters(modality)
    all_param_names = get_all_modality_params(modality)
    param_indices = [all_param_names.index(p) for p in parameters]

    y_pred = torch.stack(results['predictions']).cpu().numpy()
    y_true = torch.stack(results['y_true']).cpu().numpy()

    fig, axes = plt.subplots(1, len(parameters), figsize=(5 * len(parameters), 5))
    if len(parameters) == 1:
        axes = [axes]

    for ax, param, idx in zip(axes, parameters, param_indices):
        ax.scatter(y_true[:, idx], y_pred[:, idx], s=4, alpha=0.3, rasterized=True)
        ax.set_xlabel(f'True {param}')
        ax.set_ylabel(f'Predicted {param}')
        ax.set_title(param)

        if spearman_r is not None:
            rho = spearman_r['pred_vs_true'].get(param)
            if rho is not None:
                ax.annotate(f'ρ = {rho:.3f}', xy=(0.05, 0.92), xycoords='axes fraction', fontsize=9)

    fig.suptitle('Predicted vs True parameters')
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Scatter plot saved to {save_path}')
    else:
        plt.show()
