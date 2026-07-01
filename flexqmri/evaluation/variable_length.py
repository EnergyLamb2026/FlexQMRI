"""Adaptation utilities for the variable-length robustness experiment.

Provides undersampling (pure index selection) and interpolation (cubic spline)
to convert test signals into the fixed lengths expected by MLP and Transformer
baselines, and a helper to build a test-only DataLoader from pre-generated
numpy arrays.
"""

import copy
from typing import List, Optional

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch.utils.data import DataLoader

from flexqmri.dataset.loaders import DataLoaderFactory


def undersample_signals(X: np.ndarray, test_x_values: List[float],
                        target_x_values: List[float]) -> np.ndarray:
    """Subsample signals to a strict subset of the measured x-values (b-values or TEs).

    Args:
        X: Shape (n_samples, len(test_x_values), 2), channel 0 = x-values,
            channel 1 = signals.
        test_x_values: The x-values at which X was measured.
        target_x_values: Subset of test_x_values to keep.

    Returns:
        Shape (n_samples, len(target_x_values), 2).

    Raises:
        ValueError: If any value in target_x_values is not in test_x_values.
    """
    test_list = list(test_x_values)
    indices = []
    for x in target_x_values:
        if x not in test_list:
            raise ValueError(f"x={x} is not in test_x_values={test_x_values}")
        indices.append(test_list.index(x))
    return X[:, indices, :]


def interpolate_signals(X: np.ndarray,
                        target_x_values: Optional[List[float]] = None,
                        target_n: Optional[int] = None) -> np.ndarray:
    """Interpolate signals to target x-values using a per-sample cubic spline.

    Fits a cubic spline through the valid (non-NaN) measured (x-value, signal)
    pairs and evaluates it at the target x-values. Two targeting modes:

    - ``target_x_values``: fixed set of x-values shared across all samples
      (IVIM use case, no NaN in data).
    - ``target_n``: ``target_n`` x-values uniformly spaced between each
      sample's min and max valid x-value (T2* variable-length use case, NaN
      padding in data).

    Args:
        X: Shape (n_samples, n_measured, 2), channel 0 = x-values,
            channel 1 = signals. May contain NaN padding.
        target_x_values: Fixed x-values at which to evaluate the spline.
            Must be provided if ``target_n`` is None.
        target_n: Number of equally-spaced target points per sample, spanning
            each sample's valid x-value range. Must be provided if
            ``target_x_values`` is None.

    Returns:
        Shape (n_samples, n_target, 2). Channel 0 holds the target x-values
        (per-sample when ``target_n`` is used); channel 1 holds interpolated
        signal values.

    Raises:
        ValueError: If neither or both of ``target_x_values`` and ``target_n``
            are provided.
    """
    if (target_x_values is None) == (target_n is None):
        raise ValueError("Provide exactly one of target_x_values or target_n.")
    n_samples = X.shape[0]
    n_target = target_n if target_n is not None else len(target_x_values)
    X_interp = np.empty((n_samples, n_target, 2), dtype=np.float32)

    for i in range(n_samples):
        x_vals = X[i, :, 0].astype(np.float64)
        signals = X[i, :, 1].astype(np.float64)
        valid = ~np.isnan(signals)
        x_valid = x_vals[valid]
        s_valid = signals[valid]

        if target_n is not None:
            target_arr = np.linspace(x_valid.min(), x_valid.max(), target_n)
        else:
            target_arr = np.array(target_x_values, dtype=np.float64)

        cs = CubicSpline(x_valid, s_valid)
        X_interp[i, :, 0] = target_arr.astype(np.float32)
        X_interp[i, :, 1] = cs(target_arr).astype(np.float32)

    return X_interp


def make_test_loader(X: np.ndarray, y: np.ndarray, noise: np.ndarray,
                     model_config: dict,
                     generator: torch.Generator = None) -> DataLoader:
    """Create a test-only DataLoader from pre-generated numpy arrays.

    Overrides train_val_test_split to [0, 0, 1] so all samples go to the test
    split. Uses DataLoaderFactory so NCDE coefficient pre-computation and
    fixed-length filtering happen automatically based on model_config.

    Args:
        X: Input signals, shape (n_samples, seq_len, 2).
        y: Target parameters, shape (n_samples, n_params).
        noise: Noise levels, shape (n_samples,) or (n_samples, 1).
        model_config: Full model configuration dict loaded from config.json.
        generator: PyTorch generator for reproducibility.

    Returns:
        DataLoader containing all samples as the test split.
    """
    config = copy.deepcopy(model_config)
    config["data"]["simulation"]["train_val_test_split"] = [0.0, 0.0, 1.0]
    factory = DataLoaderFactory(config)
    _, _, test_loader = factory.create_loaders(X, y, noise, generator)
    return test_loader
