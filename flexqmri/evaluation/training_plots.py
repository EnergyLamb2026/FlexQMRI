"""Plotting functions for training monitoring and signal diagnostics."""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless/SSH environments
import numpy as np
import torch
import matplotlib.pyplot as plt

from flexqmri.evaluation import utils
from flexqmri.utils import biophysical_model

# ---------------------------------------------------------------------------
# Loss / lambda / uncertainty plots  (logged by supervised.py after training)
# ---------------------------------------------------------------------------

def plot_losses(
    config: dict,
    losses: list,
    val_losses: list,
    sup_losses: list = None,
    phys_inf_losses: list = None,
):
    """Plot the training and validation losses.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        losses (list): List of training losses.
        val_losses (list): List of validation losses.
        sup_losses (list, optional): List of supervised losses (if using physics-informed loss).
        phys_inf_losses (list, optional): List of physics-informed losses.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.title('Training and Validation Loss')
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    if config["train"]["pil"]:
        plt.plot(sup_losses, label='Supervised Loss', linewidth=2)
        plt.plot(phys_inf_losses, label='Physics Informed Loss', linewidth=2)
    if config["train"]["lambda_update_frequency"] == 0:
        plt.plot(losses, label='Training Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Signal plots
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PSD utilities
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Gradient utilities and plots
# ---------------------------------------------------------------------------

def compute_gradient(model: torch.nn.Module, optimizer, loss: torch.Tensor) -> list:
    """Compute the gradients of the model parameters for a given loss.

    Args:
        model (torch.nn.Module): The neural network model.
        optimizer: The optimizer used for training.
        loss (torch.Tensor): The loss to compute gradients for.

    Returns:
        list[torch.Tensor]: List of gradient tensors, one per parameter that has a gradient.
    """
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    return [param.grad.clone() for _, param in model.named_parameters() if param.grad is not None]

def plot_grad_histogram(
    model: torch.nn.Module,
    optimizer,
    supervised_loss: torch.Tensor,
    phys_inf_loss: torch.Tensor,
    save_path: str = None,
):
    """Plot histograms of gradients for the supervised and physics-informed losses.

    Args:
        model (torch.nn.Module): The neural network model.
        optimizer: The optimizer used for training.
        supervised_loss (torch.Tensor): The supervised loss.
        phys_inf_loss (torch.Tensor): The physics-informed loss.
        save_path (str, optional): If provided, save the figure to this path instead of showing it.

    Returns:
        None
    """
    grad_supervised = torch.cat([g.reshape(-1) for g in compute_gradient(model, optimizer, supervised_loss)])
    grad_phys_inf = torch.cat([g.reshape(-1) for g in compute_gradient(model, optimizer, phys_inf_loss)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(grad_supervised.cpu().numpy(), bins=1000, color='blue')
    axes[0].set_title('Supervised Loss Gradients')
    axes[0].set_xlabel('Gradient value')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(grad_phys_inf.cpu().numpy(), bins=1000, color='red')
    axes[1].set_title('Physics Informed Loss Gradients')
    axes[1].set_xlabel('Gradient value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim(-1e-6, 1e-6)
    axes[1].set_ylim(0, 16000)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
