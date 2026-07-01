"""
DataLoader factory for creating PyTorch DataLoaders from MRI datasets.

This module provides a unified interface for creating train/val/test loaders
for synthetic MRI datasets.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchcde

from flexqmri.networks import ncde as ncde_utils

def filter_samples_by_length(X: np.ndarray, y: np.ndarray, noise: np.ndarray,
                            target_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter samples to keep only those with a specific signal length.
    
    For variable-length data (with NaN padding), this function:
    1. Selects only samples where the number of non-NaN values matches target_length
    2. Trims X to only keep the first target_length measurements (removes NaN padding)
    
    X must be 3D (n_samples, max_length, 2) where channel 0 contains measurement
    coordinates and channel 1 contains signal values.

    Args:
        X: Input signals, shape (n_samples, max_length, 2)
        y: Target parameters, shape (n_samples, n_params)
        noise: Noise levels, shape (n_samples,) or (n_samples, 1)
        target_length: Desired signal length (number of non-NaN measurements)
        
    Returns:
        Tuple of (X_filtered, y_filtered, noise_filtered)
    """
    # Calculate actual length (non-NaN count) for each sample using the signal channel
    actual_lengths = np.sum(~np.isnan(X[:, :, 1]), axis=1)

    # Find samples with target length
    mask = actual_lengths == target_length

    # Filter and trim all arrays
    X_filtered = X[mask, :target_length, :]
    y_filtered = y[mask]
    noise_filtered = noise[mask]

    n_original = X.shape[0]
    n_filtered = X_filtered.shape[0]
    print(f"Filtered samples: {n_original} -> {n_filtered} (keeping only {target_length}-length samples)")
    print(f"Trimmed X shape: {X.shape} -> {X_filtered.shape}")
    
    return X_filtered, y_filtered, noise_filtered


class DataLoaderFactory:
    """
    Factory class for creating PyTorch DataLoaders.
    
    Handles data splitting, batching, and optional preprocessing
    (like NCDE interpolation) before creating loaders.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataLoader factory.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config["data"]
        # For accessing simulation-specific parameters
        self.simulation_config = config["data"].get("simulation", {})
        self.train_config = config["train"]
        
        # Split ratios — read from simulation config if present; default to all-test
        # for in-vivo data where train/val splits are not applicable.
        sim_config = self.data_config.get("simulation", {})
        self.train_val_test_split = sim_config.get("train_val_test_split", [0.0, 0.0, 1.0])
        
        # Batch sizes
        self.train_batch_size = self.train_config["batch_size"]
        self.test_batch_size = config["test"]["batch_size"]
        
        # Model type (for NCDE interpolation)          
        self.model_type = self.train_config["model"]
        self.interpolation_during_training = self.train_config.get("interpolation_during_training", False)
    
    def create_loaders(self, X: np.ndarray, y: np.ndarray,
                      noise: np.ndarray,
                      generator: Optional[torch.Generator] = None
                     ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test DataLoaders.
        
        For non-NCDE models with fixed_length > 0, filters to keep only
        samples with the target signal length.
        
        Args:
            X: Input signals, shape (n_samples, n_measurements, 2)
            y: Target parameters, shape (n_samples, n_params)
            noise: Noise levels, shape (n_samples, 1)
            generator: PyTorch generator for reproducible splits
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Filter samples by length for non-LSQ models with fixed_length > 0
        # (NCDE always has fixed_length = 0, so it's excluded by the condition)
        fixed_length = self.data_config.get("fixed_length", 0)
        if fixed_length > 0 and self.model_type != 'lsq':
            X, y, noise = filter_samples_by_length(X, y, noise, fixed_length)
                
        y_tensor = torch.from_numpy(y).float()
        
        # Ensure noise has shape (n_samples, 1)
        if noise.ndim == 1:
            noise = noise[:, None]
        elif noise.ndim > 2:
            noise = noise.reshape(noise.shape[0], -1)
        noise_tensor = torch.from_numpy(noise).float()
        
        # X is always 3D (n_samples, n_measurements, 2): channel 0 = coordinates, channel 1 = signals
        X_tensor = torch.from_numpy(X).float()

        # Apply NCDE interpolation if needed
        if self.model_type == 'ncde' and not self.interpolation_during_training:
            coeffs = self._apply_ncde_interpolation_from_path(X_tensor)
            # For NCDE with pre-computed coeffs: [coeffs, X, y, noise]
            dataset = TensorDataset(coeffs.float(), X_tensor, y_tensor, noise_tensor)
        else:
            # For MLP/other models or NCDE with interpolation_during_training=True: [X, y, noise]
            dataset = TensorDataset(X_tensor, y_tensor, noise_tensor)
        
        # Split dataset
        train_loader, val_loader, test_loader = self._split_and_create_loaders(
            dataset, generator
        )
        
        return train_loader, val_loader, test_loader
    
    def _apply_ncde_interpolation_from_path(self, path_tensor: torch.Tensor) -> torch.Tensor:
        """Apply NCDE interpolation when path (n, len, 2) is already built.

        Args:
            path_tensor: Pre-built path tensor, shape (n_samples, n_measurements, 2)

        Returns:
            Hermite cubic interpolation coefficients
        """
        path_prep = ncde_utils.prep_inputs(path_tensor)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(path_prep)
        return coeffs.detach().cpu()

    def _split_and_create_loaders(self, dataset: TensorDataset, 
                                  generator: Optional[torch.Generator]
                                 ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Split dataset and create DataLoaders.
        
        Args:
            dataset: TensorDataset to split
            generator: Random generator for reproducible splits
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Calculate split sizes
        n_samples = len(dataset)
        train_size = int(self.train_val_test_split[0] * n_samples)
        val_size = int(self.train_val_test_split[1] * n_samples)
        test_size = n_samples - train_size - val_size

        # Skip random_split when all samples go to test (e.g. in-vivo inference):
        # random_split internally calls randperm which would permute the indices
        # even with a [0, 0, 1] split, destroying the original spatial ordering.
        if train_size == 0 and val_size == 0:
            train_dataset, val_dataset, test_dataset = [], [], dataset
        else:
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size], generator=generator
            )

        # Create DataLoaders — shuffle only when non-empty (RandomSampler rejects size 0)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=len(train_dataset) > 0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.train_batch_size,
            shuffle=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False
        )

        return train_loader, val_loader, test_loader
