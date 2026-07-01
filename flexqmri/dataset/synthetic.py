"""
Synthetic MRI dataset implementations.

This module contains concrete implementations of synthetic dataset generators
for different MRI modalities (IVIM, T2*, etc.).
"""

from typing import Tuple
import numpy as np
import torch

from .base import DatasetMRSynth


class SynthIVIM(DatasetMRSynth):
    """
    Synthetic IVIM (Intravoxel Incoherent Motion) dataset generator.
    
    Generates synthetic IVIM signals using the biexponential model:
    S(b) = S0 * [f * exp(-b * D*) + (1-f) * exp(-b * D)]
    
    Parameters:
        - S0: Signal at b=0
        - f: Perfusion fraction
        - D: Diffusion coefficient (mm²/s)
        - D*: Pseudo-diffusion coefficient (mm²/s)
    """
    
    def __init__(self, config):
        """
        Initialize IVIM synthetic dataset.
        
        Args:
            config: Configuration dictionary with data and train sections
        """
        super().__init__(config, modality='ivim')
        
        # IVIM-specific parameters
        self.b_values_set = np.array(self.data_config["x"])
        self.fixed_length = config["data"].get("fixed_length", 0)
        self.min_b_values = self.data_config["min_x_length"]
        self.max_b_values = self.data_config["max_x_length"]
        self.b_jitter = self.data_config.get("b_jitter", 0.0)  # Relative std for b-value perturbation
        
        if len(self.b_values_set) == 0:
            raise ValueError("B-values (x) must be provided in config['data']['x']")
        
        # Validate param_ranges has 4 parameters
        if len(self.param_ranges) != 4:
            raise ValueError(f"IVIM requires 4 parameter ranges [S0, f, D, D*], got {len(self.param_ranges)}")
    
    def _generate_parameters(self, random_generator: torch.Generator) -> np.ndarray:
        """
        Generate random IVIM parameters within specified ranges.
        
        Args:
            random_generator: PyTorch generator for reproducibility
            
        Returns:
            Normalized coefficients in [0, 1], shape (n_samples, 4) - [S0, f, D, D*]
        """
        coeffs = torch.rand(self.n_samples, 4, generator=random_generator).numpy()
        self.param_coeffs = coeffs
        return coeffs
    
    def _generate_signals(self, parameters: np.ndarray,
                         random_generator: torch.Generator
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic IVIM signals from parameters.
        
        Args:
            parameters: Normalized coefficients in [0, 1], shape (n_samples, 4)
            random_generator: PyTorch generator
            
        Returns:
            Tuple of (X, noise_levels) where:
                - X: shape (n_samples, n_b_values, 2) - channel 0 = b-values, channel 1 = signals
                - noise_levels: shape (n_samples, 1) - SNR values
        """
        physical = self.rescale_coeffs(parameters)
        S0 = physical[:, 0][:, np.newaxis]
        f = physical[:, 1][:, np.newaxis]
        D = physical[:, 2][:, np.newaxis]
        D_star = physical[:, 3][:, np.newaxis]
        
        b_values_subset = self._sample_b_values(random_generator) # Determine which b-values to use
        b_values_subset = self._perturb_b_values(b_values_subset, random_generator)  # Add realistic scanner jitter
        
        signals_clean = self._ivim_biexp_model(b_values_subset, S0, f, D, D_star) # Generate clean IVIM signals: S(b) = S0 * [f * exp(-b * D*) + (1-f) * exp(-b * D)]
        
        signals_noisy, noise_levels = self._add_uniform_noise(signals_clean, random_generator)
        X = np.stack([b_values_subset, signals_noisy], axis=2)
        return X, noise_levels

    def _ivim_biexp_model(self, b_values: np.ndarray, S0: np.ndarray,
                          f: np.ndarray, D: np.ndarray, D_star: np.ndarray
                         ) -> np.ndarray:
        """
        IVIM biexponential model.
        
        Args:
            b_values: B-values, shape (n_samples, n_b) or (1, n_b)
            S0: Signal at b=0, shape (n_samples, 1)
            f: Perfusion fraction, shape (n_samples, 1)
            D: Diffusion coefficient, shape (n_samples, 1)
            D_star: Pseudo-diffusion coefficient, shape (n_samples, 1)
            
        Returns:
            Signals, shape (n_samples, n_b_values)
        """
        perfusion_term = f * np.exp(-b_values * D_star)
        diffusion_term = (1 - f) * np.exp(-b_values * D)
        signals = S0 * (perfusion_term + diffusion_term)
        
        return signals
    
    def _sample_b_values(self, random_generator: torch.Generator) -> np.ndarray:
        """
        Sample b-values from intervals: 0, (0-50), (50-100), (100-800), 800.
        
        This interval-based sampling ensures:
        - Always includes b=0 (smallest)
        - Always includes b=800 (highest)
        - Distributes remaining samples across intervals based on fixed_length or random between min/max
        - Acts as data augmentation
        
        Args:
            random_generator: PyTorch generator
            
        Returns:
            B-values subset as 2D array, shape (n_samples, len(self.b_values_set))
            Non-selected positions are filled with NaN
        """
        b_values_array_length = len(self.b_values_set)
        
        if self.fixed_length > 0:
            return self._sample_fixed_length_b_values(b_values_array_length, random_generator)
        else:
            return self._sample_variable_length_b_values(b_values_array_length, random_generator)
    
    def _sample_fixed_length_b_values(self, array_length: int, random_generator: torch.Generator) -> np.ndarray:
        """Sample fixed-length b-values for all samples."""
        b_values_sampled = np.full((self.n_samples, array_length), np.nan)
        for i in range(self.n_samples):
            b_vals = self._sample_interval_b_values(self.fixed_length, random_generator)
            b_values_sampled[i, :len(b_vals)] = b_vals
        return b_values_sampled
    
    def _sample_variable_length_b_values(self, array_length: int, random_generator: torch.Generator) -> np.ndarray:
        """Sample variable-length b-values for each sample."""
        b_values_sampled = np.full((self.n_samples, array_length), np.nan)
        
        for i in range(self.n_samples):
            length = int(torch.randint(self.min_b_values, self.max_b_values + 1, (1,), generator=random_generator).item())
            b_vals = self._sample_interval_b_values(length, random_generator)
            b_values_sampled[i, :len(b_vals)] = b_vals
        
        return b_values_sampled
    
    def _sample_interval_b_values(self, n_measurements: int, 
                                   random_generator: torch.Generator) -> np.ndarray:
        """
        Sample b-values from intervals following the old implementation logic.
        
        Distribution strategy (following get_time_indices):
        - Always include b=0 (smallest)
        - Always include b=800 (highest)
        - Remaining slots distributed across intervals:
          - Below 50 (b < 50)
          - Between 50-100 (50 <= b < 100)
          - Above 100 (100 <= b < 800)
        
        Args:
            n_measurements: Total number of b-values to sample
            random_generator: PyTorch generator
            
        Returns:
            Sorted b-values array of length n_measurements
        """
        # Always keep b=0 and b=800 (2 measurements reserved)
        remaining_measurements = n_measurements - 2
        
        if remaining_measurements < 0:
            # Edge case: only room for b=0 or b=800
            if n_measurements == 1:
                return np.array([0.0])
            else:
                return np.array([0.0, 800.0])
        
        # Distribute remaining measurements across intervals
        size_below_bval100 = int(np.ceil(remaining_measurements / 2))
        size_below_bval50 = int(np.ceil(size_below_bval100 / 2))
        size_between_bval_50_100 = size_below_bval100 - size_below_bval50
        size_above_bval100 = remaining_measurements - size_below_bval100
        
        # Get indices for each interval
        below_50_idcs = np.where((self.b_values_set > 0) & (self.b_values_set < 50))[0]
        between_50_100_idcs = np.where((self.b_values_set >= 50) & (self.b_values_set < 100))[0]
        above_100_idcs = np.where((self.b_values_set >= 100) & (self.b_values_set < 800))[0]
        
        # Randomly select from each interval
        b_vals_list = [0]  # Always include b=0
        
        # Sample from below 50
        if len(below_50_idcs) > 0 and size_below_bval50 > 0:
            sample_size = min(size_below_bval50, len(below_50_idcs))
            sampled_idcs = torch.randperm(len(below_50_idcs), generator=random_generator)[:sample_size].numpy()
            b_vals_list.extend(self.b_values_set[below_50_idcs[sampled_idcs]])
        
        # Sample from between 50-100
        if len(between_50_100_idcs) > 0 and size_between_bval_50_100 > 0:
            sample_size = min(size_between_bval_50_100, len(between_50_100_idcs))
            sampled_idcs = torch.randperm(len(between_50_100_idcs), generator=random_generator)[:sample_size].numpy()
            b_vals_list.extend(self.b_values_set[between_50_100_idcs[sampled_idcs]])
        
        # Sample from above 100
        if len(above_100_idcs) > 0 and size_above_bval100 > 0:
            sample_size = min(size_above_bval100, len(above_100_idcs))
            sampled_idcs = torch.randperm(len(above_100_idcs), generator=random_generator)[:sample_size].numpy()
            b_vals_list.extend(self.b_values_set[above_100_idcs[sampled_idcs]])
        
        # Always include b=800 (highest)
        b_vals_list.append(800.0)
        
        # Ensure we have exactly n_measurements by removing duplicates and truncating
        b_vals = np.array(sorted(set(b_vals_list)))
        
        # If we have fewer than needed, fill with random values from all intervals
        while len(b_vals) < n_measurements:
            # Sample from all available b-values (excluding 0 and 800 already added)
            candidates = self.b_values_set[(self.b_values_set > 0) & (self.b_values_set < 800)]
            rand_idx = int(torch.randint(0, len(candidates), (1,), generator=random_generator).item())
            new_val = candidates[rand_idx]
            if new_val not in b_vals:
                b_vals = np.array(sorted(set(list(b_vals) + [new_val])))
        
        # Truncate to exact length
        return b_vals[:n_measurements]
    
    def _perturb_b_values(self, b_values: np.ndarray,
                           random_generator: torch.Generator) -> np.ndarray:
        """
        Apply small Gaussian jitter to non-zero b-values to mimic MRI scanner imprecision.

        The perturbation is relative to each nominal b-value:
            b_perturbed = clip(b + N(0, (b_jitter * b)^2), min=1)

        b=0 entries are always left unchanged (they anchor S(b=0) = S0).
        NaN-padded slots are preserved.

        Args:
            b_values: B-values array, shape (n_samples, n_b_values), may contain NaN padding.
            random_generator: PyTorch generator for reproducibility.

        Returns:
            Perturbed b-values, same shape as input.
        """
        if self.b_jitter == 0.0:
            return b_values

        perturbed = b_values.copy()
        non_zero_mask = (b_values > 0) & ~np.isnan(b_values)
        std = self.b_jitter * b_values[non_zero_mask]
        noise_tensor = torch.randn(int(non_zero_mask.sum()), generator=random_generator).numpy()
        perturbed[non_zero_mask] = np.clip(b_values[non_zero_mask] + noise_tensor * std, a_min=1.0, a_max=None)
        return perturbed

    def _add_uniform_noise(self, signals_clean: np.ndarray,
                           random_generator: torch.Generator
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Add uniform noise scaled by SNR using S0 from signal at b=0."""
        s0_values = signals_clean[:, 0]  # Signal at b=0 equals S0
        return super()._add_uniform_noise(signals_clean, s0_values, random_generator)


class SynthT2Star(DatasetMRSynth):
    """Synthetic T2* dataset generator using mono-exponential decay.

    Generates synthetic T2* signals using the model:
    S(TE) = S0 * exp(-TE / T2*)

    Each sample gets a random set of echo times (TEs) drawn uniformly
    from a continuous range, simulating realistic clinical acquisition
    variability.

    Parameters:
        - S0: Signal intensity at TE=0
        - T2*: Effective transverse relaxation time (ms)
    """

    def __init__(self, config):
        """Initialize T2* synthetic dataset.

        Args:
            config: Configuration dictionary with data and train sections
        """
        super().__init__(config, modality='t2star')

        # T2*-specific parameters
        self.te_range = np.array(self.data_config["x_range"], dtype=float)
        self.fixed_length = config["data"].get("fixed_length", 0)
        self.min_te_values = self.data_config["min_x_length"]
        self.te_values_set = np.array(self.data_config["x"], dtype=float) if "x" in self.data_config else None
        self.max_te_values = len(self.te_values_set) if self.te_values_set is not None else self.data_config["max_x_length"]

        if self.te_range[0] >= self.te_range[1]:
            raise ValueError(
                f"te_range[0] must be < te_range[1], got {self.te_range}"
            )
        if len(self.param_ranges) != 2:
            raise ValueError(
                f"T2* requires 2 parameter ranges [S0, T2*], got {len(self.param_ranges)}"
            )

    def _generate_parameters(self, random_generator: torch.Generator) -> np.ndarray:
        """Generate random T2* parameters within specified ranges.

        Args:
            random_generator: PyTorch generator for reproducibility

        Returns:
            Normalized coefficients in [0, 1], shape (n_samples, 2) - [S0, T2*]
        """
        coeffs = torch.rand(self.n_samples, 2, generator=random_generator).numpy()
        self.param_coeffs = coeffs
        return coeffs

    def _generate_signals(self, parameters: np.ndarray,
                         random_generator: torch.Generator
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic T2* signals from parameters.

        Returns X as a 3D array of shape (n_samples, max_te_values, 2) where
        channel 0 contains echo times and channel 1 contains signal values.
        Non-selected positions are NaN-padded.

        Args:
            parameters: Normalized coefficients in [0, 1], shape (n_samples, 2)
            random_generator: PyTorch generator

        Returns:
            Tuple of (X, noise_levels) where:
                - X: shape (n_samples, max_te_values, 2)
                - noise_levels: shape (n_samples, 1)
        """
        physical = self.rescale_coeffs(parameters)
        S0 = physical[:, 0]
        T2star = physical[:, 1]

        te_array = self._sample_te_values(random_generator)
        signals_clean = self._t2star_mono_exp_model(te_array, S0, T2star)
        signals_noisy, noise_levels = self._add_noise(signals_clean, S0, random_generator)

        # Pack TEs and signals into 3D array (n_samples, max_te, 2)
        X = np.stack([te_array, signals_noisy], axis=2)

        return X, noise_levels

    def _t2star_mono_exp_model(self, te_array: np.ndarray,
                               S0: np.ndarray, T2star: np.ndarray) -> np.ndarray:
        """T2* mono-exponential decay: S(TE) = S0 * exp(-TE / T2*).

        Args:
            te_array: Echo times, shape (n_samples, max_te_values), may contain NaN
            S0: Signal at TE=0, shape (n_samples,)
            T2star: T2* values, shape (n_samples,)

        Returns:
            Signals, shape (n_samples, max_te_values)
        """
        S0_col = S0[:, np.newaxis]
        T2star_col = T2star[:, np.newaxis]
        return S0_col * np.exp(-te_array / T2star_col)

    def _sample_te_values(self, random_generator: torch.Generator) -> np.ndarray:
        """Sample random echo times for each sample from a continuous range.

        Each sample gets a random number of TEs (between min_te_values and
        max_te_values, or fixed_length if > 0). TEs are drawn uniformly from
        [te_range[0], te_range[1]] and sorted in ascending order. Unused
        positions are NaN-padded.

        Args:
            random_generator: PyTorch generator

        Returns:
            TE array, shape (n_samples, max_te_values), NaN-padded
        """
        if self.te_values_set is not None:
            return np.tile(self.te_values_set, (self.n_samples, 1)).astype(np.float32)

        te_array = np.full((self.n_samples, self.max_te_values), np.nan)

        for i in range(self.n_samples):
            n_te = self._get_n_te(random_generator)
            te_values = self.te_range[0] + (self.te_range[1] - self.te_range[0]) * torch.rand(
                n_te, generator=random_generator
            ).numpy()
            te_values = np.sort(te_values)
            te_array[i, :n_te] = te_values

        return te_array

    def _get_n_te(self, random_generator: torch.Generator) -> int:
        """Get the number of TEs for a sample (fixed or random).

        Args:
            random_generator: PyTorch generator

        Returns:
            Number of echo times
        """
        if self.fixed_length > 0:
            return self.fixed_length
        return int(torch.randint(
            self.min_te_values, self.max_te_values + 1, (1,),
            generator=random_generator
        ).item())

    def _add_noise(self, signals_clean: np.ndarray, s0_values: np.ndarray,
                   random_generator: torch.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Add uniform noise scaled by SNR.

        Args:
            signals_clean: Clean signals, shape (n_samples, max_te_values)
            s0_values: S0 parameter values, shape (n_samples,)
            random_generator: PyTorch generator

        Returns:
            Tuple of (noisy_signals, snr_values)
        """
        return super()._add_uniform_noise(signals_clean, s0_values, random_generator)
