"""
In-vivo MRI dataset wrapper for real patient data.
"""

from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch

from .base import DatasetMR


class DatasetMRReal(DatasetMR):
    """In-vivo MRI dataset wrapping pre-loaded patient signals.

    A single class handles all modalities (IVIM, T2*, etc.) because real data
    loading has no modality-specific logic. The synthetic counterparts
    (SynthIVIM, SynthT2Star) are separate classes only because they implement
    different biophysical forward models for signal generation — a distinction
    that does not exist here. The modality is still stored and forwarded for
    downstream use (e.g. LSQ fitting, rescaling).

    Args:
        config (dict): Configuration dictionary with 'data' (including an
            'invivo' subsection with 'x' and 'n_samples') and 'train' sections.
        modality (str): MRI modality ('ivim' or 't2star').
        X (np.ndarray): Raw signal array, shape (n_samples, n_measurements).
    """

    def __init__(self, config: Dict[str, Any], modality: str, X: np.ndarray):
        """Initialize in-vivo MRI dataset.

        Args:
            config (dict): Configuration dictionary. Must contain
                'data.invivo.x' (measurement coordinates) and 'data.invivo.n_samples'.
            modality (str): MRI modality ('ivim' or 't2star').
            X (np.ndarray): Signal array, shape (n_samples, n_measurements).
        """
        super().__init__(config, modality)
        self._X_raw = np.asarray(X, dtype=np.float32)
        self.invivo_config = self.data_config['invivo']
        self.x = np.asarray(self.invivo_config['x'], dtype=np.float32)
        self._validate_config()

    def _validate_config(self):
        pass

    def generate_data(self, random_generator: Optional[torch.Generator] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Format in-vivo signals into the (n_samples, n_meas, 2) path format.

        Stacks measurement coordinates (b-values or echo times) as channel 0
        and signal values as channel 1, matching the format produced by the
        synthetic dataset classes.

        Ground-truth parameters (y) and noise levels are unavailable for real
        data, so zero arrays are returned as placeholders. They are never used
        during inference, only during training where ground truth is required.

        Args:
            random_generator (torch.Generator, optional): Unused; kept for
                interface compatibility with DatasetMR.

        Returns:
            X (np.ndarray): Shape (n_samples, n_meas, 2).
                Channel 0: measurement coordinates (b-values / TEs).
                Channel 1: signal values.
            y (np.ndarray): Zero placeholder, shape (n_samples, 1).
            noise (np.ndarray): Zero placeholder, shape (n_samples, 1).
        """
        n_samples, n_meas = self._X_raw.shape

        coords = np.tile(self.x, (n_samples, 1))          # (n_samples, n_meas)
        self.X = np.stack([coords, self._X_raw], axis=2)  # (n_samples, n_meas, 2)
        self.y = np.zeros((n_samples, 1), dtype=np.float32)
        self.noise = np.zeros((n_samples, 1), dtype=np.float32)

        return self.X, self.y, self.noise
