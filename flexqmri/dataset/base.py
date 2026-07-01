"""
Base classes for MRI dataset management.

This module provides abstract base classes for handling MRI regression datasets.
All dataset types (synthetic) inherit from these base classes.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from flexqmri.utils.biophysical_model import rescale_coeffs_torch, get_model_from_config


class DatasetMR(ABC):
    """
    Abstract base class for all MRI regression datasets.
    
    Defines the common interface for synthetic datasets.
    """
    
    def __init__(self, config: Dict[str, Any], modality: str):
        """
        Initialize MRI dataset.
        
        Args:
            config (dict): Configuration dictionary with 'data' and 'train' sections
            modality (str): MRI modality (e.g., 'ivim')
        """
        self.config = config
        self.modality = modality.lower()
        self.data_config = config["data"]
        self.train_config = config["train"]
        
        # Data storage
        self.X = None
        self.y = None
        self.noise = None
        self.param_coeffs = None
    
    @abstractmethod
    def _validate_config(self):
        """Validate that required config parameters are present."""
        pass
    
    @abstractmethod
    def generate_data(self, random_generator: Optional[torch.Generator] = None
                     ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate or load the dataset.
        
        Args:
            random_generator: PyTorch generator for reproducibility
            
        Returns:
            Tuple of (X, y, noise) where:
                - X: Input signals, shape (n_samples, n_measurements)
                - y: Target parameters, shape (n_samples, n_params)
                - noise: Noise levels (optional), shape (n_samples,)
        """
        pass
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get the generated/loaded data."""
        if self.X is None:
            raise ValueError("Data not yet generated. Call generate_data() first.")
        return self.X, self.y, self.noise
    
    def get_modality(self) -> str:
        """Get the modality of this dataset."""
        return self.modality

    def rescale_coeffs(self, coeffs: np.ndarray,
                       param_ranges: Optional[np.ndarray] = None) -> np.ndarray:
        """Rescale unit-range coefficients to parameter values using configured ranges.

        Delegates to ``utils.phys_models.rescale_coeffs`` (torch-based) and
        converts back to numpy.

        Args:
            coeffs (np.ndarray): Coefficients in [0, 1] range.
            param_ranges (np.ndarray, optional): Overrides ``self.param_ranges``.

        Returns:
            np.ndarray: Rescaled parameter values.
        """
        ranges = (param_ranges if param_ranges is not None else self.param_ranges).tolist()
        coeffs_t = torch.from_numpy(np.asarray(coeffs, dtype=np.float32))
        return rescale_coeffs_torch(ranges, coeffs_t).numpy()


class DatasetMRSynth(DatasetMR):
    """
    Abstract base class for synthetic MRI datasets.
    
    Provides common functionality for synthetic data generation,
    including save/load, parameter sampling, and SNR control.
    """
    
    def __init__(self, config: Dict[str, Any], modality: str):
        """Initialize synthetic MRI dataset."""
        super().__init__(config, modality)
        
        # Override data_config to point to simulation subsection
        self.data_config = config["data"]["simulation"]
        
        # Common synthetic data parameters
        self.n_samples = self.data_config["n_samples"]
        self.snr_range = np.asarray(self.data_config["snr_range"], dtype=float)
        self.param_ranges = np.asarray(
            get_model_from_config(config).param_ranges, dtype=float
        )
        self.load_data = self.data_config["load"]
        self.save_data = self.data_config["save"]
        self.input_file = self.data_config["input_file"]
        
        # Validate synthetic config
        self._validate_config()
    
    def _validate_config(self):
        """Validate synthetic data configuration."""
        required_keys = [
            'snr_range',
            'n_samples',
            'load',
            'save',
            'input_file'
        ]
        for key in required_keys:
            if key not in self.data_config:
                raise ValueError(f"Missing required config key in simulation section: {key}")
        
        # param_ranges is derived from the biophysical model, not stored in config

    def _sample_snr_log_uniform(self, random_generator: torch.Generator) -> np.ndarray:
        """Sample SNR values uniformly in logarithmic space.

        Args:
            random_generator: PyTorch generator for reproducibility

        Returns:
            SNR values, shape (n_samples,)
        """
        log_snr_min = np.log(self.snr_range[0])
        log_snr_max = np.log(self.snr_range[1])

        log_snr = log_snr_min + (log_snr_max - log_snr_min) * torch.rand(
            self.n_samples, generator=random_generator
        ).numpy()
        return np.exp(log_snr)

    def _add_uniform_noise(self, signals_clean: np.ndarray,
                           s0_values: np.ndarray,
                           random_generator: torch.Generator
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Add uniform noise scaled by SNR.

        Args:
            signals_clean: Clean signals, shape (n_samples, n_measurements)
            s0_values: S0 parameter values, shape (n_samples,)
            random_generator: PyTorch generator for reproducibility

        Returns:
            Tuple of (noisy_signals, snr_values)
        """
        snr_values = self._sample_snr_log_uniform(random_generator)
        sigma = s0_values / snr_values

        signals_tensor = torch.from_numpy(signals_clean).float()
        sigma_tensor = torch.from_numpy(sigma).float().unsqueeze(1)

        uniform_noise = (
            torch.rand(signals_tensor.shape, generator=random_generator)
            - 0.5
        ) * 2.0 * sigma_tensor
        signals_noisy = signals_tensor + uniform_noise

        return signals_noisy.numpy(), snr_values[:, np.newaxis]
    
    @abstractmethod
    def _generate_parameters(self, random_generator: torch.Generator
                            ) -> np.ndarray:
        """
        Generate random parameters within specified ranges.
        
        Args:
            random_generator: PyTorch generator
            
        Returns:
            Parameters array, shape (n_samples, n_params)
        """
        pass
    
    @abstractmethod
    def _generate_signals(self, parameters: np.ndarray, 
                         random_generator: torch.Generator
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic signals from parameters.
        
        Args:
            parameters: Ground truth parameters
            random_generator: PyTorch generator
            
        Returns:
            Tuple of (signals, noise_levels)
        """
        pass
    
    def generate_data(self, random_generator: Optional[torch.Generator] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic data or load from file.
        
        Args:
            random_generator: PyTorch generator for reproducibility
            
        Returns:
            Tuple of (X, y, noise)
        """
        if random_generator is None:
            random_generator = torch.Generator()
        
        # Try to load existing data
        if self.load_data:
            try:
                loaded_data = self._load_from_file()
                if loaded_data is not None:
                    self.X, self.y, self.noise = loaded_data
                    print(f"Loaded synthetic {self.modality.upper()} data from {self.input_file}")
                    return self.X, self.y, self.noise
            except Exception as e:
                print(f"Could not load data: {e}. Generating new data...")
        
        # Generate new data
        print(f"Generating {self.n_samples} synthetic {self.modality.upper()} samples...")
        self.y = self._generate_parameters(random_generator)
        self.X, self.noise = self._generate_signals(self.y, random_generator)
        
        # Save if requested
        if self.save_data:
            self._save_to_file()
            print(f"Saved synthetic data to {self.input_file}")
        
        return self.X, self.y, self.noise
    
    def _load_from_file(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load data from NPZ file."""
        if not Path(self.input_file).exists():
            return None
        
        data = np.load(self.input_file)
        return data['X'], data['y'], data['noise']
    
    def _save_to_file(self):
        """Save data to NPZ file."""
        np.savez(self.input_file, X=self.X, y=self.y, noise=self.noise)
