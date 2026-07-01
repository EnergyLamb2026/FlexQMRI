"""
Factory functions for creating MRI dataset instances and DataLoaders.
"""

from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import DatasetMR
from .loaders import DataLoaderFactory
from .real import DatasetMRReal
from .synthetic import SynthIVIM, SynthT2Star


def get_modality_and_data_type(config: dict) -> Tuple[str, str]:
    """Determine modality and data_type from config.

    Checks which dataset to use based on n_samples availability:
    - If config['data']['simulation']['n_samples'] > 0 -> 'simulation'
    - If config['data']['invivo']['n_samples'] > 0 -> 'invivo'
    - Modality is determined from config['data']['modality']

    Args:
        config (dict): Configuration dictionary with 'data' section.

    Returns:
        Tuple[str, str]: (modality, data_type).

    Raises:
        ValueError: If no valid data configuration is found.
    """
    modality = config['data']['modality'].lower()

    sim_n_samples = config['data'].get('simulation', {}).get('n_samples', 0)
    invivo_n_samples = config['data'].get('invivo', {}).get('n_samples', 0)

    if sim_n_samples > 0:
        data_type = 'simulation'
    elif invivo_n_samples > 0:
        data_type = 'invivo'
    else:
        raise ValueError(
            "No valid data configuration found. "
            "Set config['data']['simulation']['n_samples'] > 0 for synthetic data "
            "or config['data']['invivo']['n_samples'] > 0 for in-vivo data."
        )

    return modality, data_type


def create_dataset_instance(config: dict, modality: str, data_type: str,
                             X: Optional[np.ndarray] = None) -> DatasetMR:
    """Create appropriate dataset instance.

    Args:
        config (dict): Configuration dictionary.
        modality (str): MRI modality ('ivim' or 't2star').
        data_type (str): 'simulation' or 'invivo'.
        X (np.ndarray, optional): Raw signal array, shape (n_samples, n_meas).
            Required when data_type is 'invivo'.

    Returns:
        DatasetMR: Dataset instance.

    Raises:
        ValueError: If modality or data_type is not supported, or X is missing
            for invivo data.
    """
    modality = modality.lower()
    data_type = data_type.lower()

    if data_type == 'simulation':
        if modality == 'ivim':
            return SynthIVIM(config)
        elif modality == 't2star':
            return SynthT2Star(config)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    elif data_type == 'invivo':
        if X is None:
            raise ValueError("X must be provided when data_type is 'invivo'.")
        return DatasetMRReal(config, modality, X)
    else:
        raise ValueError(f"Unsupported data type: {data_type}. Must be 'simulation' or 'invivo'.")


def get_dataset_loaders(config: dict,
                        generator: Optional[torch.Generator] = None,
                        X: Optional[np.ndarray] = None,
                        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders for MRI regression.

    For synthetic data, X is generated internally from the config.
    For in-vivo data, X must be provided as a pre-loaded signal array of
    shape (n_samples, n_measurements).

    Args:
        config (dict): Configuration dictionary with 'data' and 'train' sections.
        generator (torch.Generator, optional): PyTorch generator for reproducibility.
        X (np.ndarray, optional): Raw signal array for in-vivo data,
            shape (n_samples, n_measurements).

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader).
            For in-vivo data the train and val loaders are empty (split [0, 0, 1]).

    Raises:
        ValueError: If modality or data_type is not supported.
    """
    # TODO: for semi-supervised we will need two separate dataset instances combined into one ConcatDataset
    if X is not None:
        data_type = 'invivo'
        modality = config['data']['modality'].lower()
    else:
        modality, data_type = get_modality_and_data_type(config)
    dataset = create_dataset_instance(config, modality, data_type, X=X)
    X_out, y, noise = dataset.generate_data(generator)
    loader_factory = DataLoaderFactory(config)
    return loader_factory.create_loaders(X_out, y, noise, generator)


