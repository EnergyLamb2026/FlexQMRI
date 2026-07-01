"""Data structures for MRI patient data handling."""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np

import flexqmri.imaging.config as load_config


@dataclass
class PatientData:
    """
    Container for patient MRI data with associated metadata.
    
    Attributes
    ----------
    data : np.ndarray
        The MRI signal data. Shape depends on modality (e.g., (n_samples, length) for flattened,
        or (x, y, z, measurements) for 4D)
    patient_id : str
        Unique patient identifier (e.g., "P001")
    study : str
        Study identifier (e.g., "Study1", "Baseline")
    serie_name : str
        Series name (e.g., "IVIM_DWI_series1")
    original_shape : tuple
        Original shape of the data before any flattening/reshaping (e.g., (64, 64, 32, 20))
    x : Optional[List[float]]
        Measurement coordinates depending on modality (e.g., b-values in s/mm² for IVIM,
        echo times in ms for T2*). Can be loaded from offset_path if not provided.
    offset_path : Optional[str]
        Path to file containing offsets/b-values. Used to load x if x is None.
    affine : Optional[np.ndarray]
        Affine transformation matrix from NIfTI file (4x4 matrix). Default None (uses identity).
    B0 : Optional[float]
        Static magnetic field strength in Tesla (e.g., 3.0). Default None.
    mask_array : Optional[np.ndarray]
        Binary mask indicating valid voxels. Shape should match spatial dimensions. Default None.
    pixel_spacing : Optional[np.ndarray]
        MRI pixel/voxel spacing in mm. Shape is (ndim,) where ndim is number of spatial dimensions.
        Usually extracted from NIfTI header. Default None.
    slice_idx : Optional[int]
        If provided, extract only this slice from 4D data during initialization.
    """
    data: np.ndarray
    patient_id: str
    study: str
    serie_name: str
    original_shape: tuple
    x: Optional[List[float]] = None
    offset_path: Optional[str] = None
    affine: Optional[np.ndarray] = None
    B0: Optional[float] = None
    mask_array: Optional[np.ndarray] = None
    pixel_spacing: Optional[np.ndarray] = None
    slice_idx: Optional[int] = None
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.data is None or not isinstance(self.data, np.ndarray):
            raise ValueError("data must be a non-None numpy array")
        if not isinstance(self.original_shape, tuple) or len(self.original_shape) == 0:
            raise ValueError("original_shape must be a non-empty tuple")
        
        # Load offsets from file if x is not provided but offset_path is
        if self.x is None and self.offset_path is not None:
            self.x = load_config.read_offsets_from_txt(self.offset_path)
            print(f"Loaded {len(self.x)} offsets from file: {self.x}")
        
        # Auto-extract slice if slice_idx is provided
        if self.slice_idx is not None:
            self._extract_slice(self.slice_idx)
    
    def get_n_samples(self) -> int:
        """Get the number of valid voxels.

        Returns the number of True entries in the mask when one is present,
        otherwise the product of all spatial dimensions (all axes except the
        last, which is the measurement axis).

        Returns:
            int: Number of valid voxels.

        Raises:
            ValueError: If data has fewer than 2 dimensions.
        """
        if self.data.ndim < 2:
            raise ValueError("Cannot determine n_samples from data shape")
        if self.mask_array is not None:
            return int(np.sum(self.mask_array > 0))
        return int(np.prod(self.data.shape[:-1]))
    
    def get_length(self) -> int:
        """Get the number of measurements per voxel.

        Returns:
            int: Size of the last (measurement) axis.

        Raises:
            ValueError: If data has fewer than 1 dimension.
        """
        if self.data.ndim < 1:
            raise ValueError("Cannot determine length from data shape")
        return self.data.shape[-1]
    
    def has_mask(self) -> bool:
        """Check if mask_array is provided."""
        return self.mask_array is not None
    
    def update_data(self, new_data: np.ndarray) -> 'PatientData':
        """
        Update the data attribute with new data (e.g., after denoising).
        
        Parameters
        ----------
        new_data : np.ndarray
            New data to replace the existing data
            
        Returns
        -------
        PatientData
            Self for method chaining
        """
        self.data = new_data
        return self
    
    def _extract_slice(self, slice_idx: int) -> 'PatientData':
        """Extract a single slice from the data along the z-axis.

        Modifies data and mask_array in place to contain only the specified
        slice.  Works for both 3D (x, y, z) and 4D (x, y, z, n) arrays.

        Args:
            slice_idx (int): Index of the slice to extract along the z-axis.

        Returns:
            PatientData: Self for method chaining.

        Raises:
            ValueError: If data has fewer than 3 dimensions.
            ValueError: If slice_idx is out of bounds for the z-axis.
        """
        if self.data.ndim < 3:
            raise ValueError(f"_extract_slice requires at least 3D data, got {self.data.ndim}D")

        z_size = self.data.shape[2]
        if slice_idx < 0 or slice_idx >= z_size:
            raise ValueError(f"slice_idx {slice_idx} out of bounds for z-axis size {z_size}")

        if self.data.ndim == 4:
            self.data = self.data[:, :, slice_idx:slice_idx+1, :]
        else:
            self.data = self.data[:, :, slice_idx:slice_idx+1]

        if self.mask_array is not None:
            self.mask_array = self.mask_array[:, :, slice_idx:slice_idx+1]

        print(f"Extracted slice {slice_idx}: data shape {self.data.shape}")
        return self
