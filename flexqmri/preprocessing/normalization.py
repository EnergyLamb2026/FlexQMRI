"""Functions to normalize multi-offset data."""

import numpy as np


def normalize_4D_stack(
    img: np.ndarray,
    norm_type: str,
) -> np.ndarray:
    """Normalize a 4D MRI image using the given method.

    Args:
        img: 4D array (x, y, z, measurements).
        norm_type: Normalization method: 'm0', 'z_score', 'minmax', or None.

    Returns:
        Normalized 4D image with the same shape as *img*.

    Raises:
        ValueError: If *norm_type* is not a supported method.

    Notes:
        'm0' normalization assumes the first offset volume is the M0 reference.
        'z_score' and 'minmax' normalize globally across all voxels and offsets.
    """
    if norm_type == "m0":
        m0 = img[:, :, :, 0:1].copy()
        m0[m0 == 0] = 1e-6
        return img / m0

    if norm_type == "z_score":
        return (img - np.mean(img)) / np.std(img)

    if norm_type == "minmax":
        return (img - np.min(img)) / (np.max(img) - np.min(img))

    if norm_type is None:
        return img

    raise ValueError(f"Normalization method '{norm_type}' not supported.")
