"""Denoising methods for multi-volume MRI images.

Includes PCA-based denoising (adaptive, MPPCA) and non-local means (NLM).

References:
    Breitling J, Deshmane A, Goerke S, et al. Adaptive denoising for MR imaging.
    NMR in Biomedicine. 2019; 32:e4133. https://doi.org/10.1002/nbm.4133
"""

from typing import Optional, Tuple

import numpy as np
from dipy.denoise.localpca import mppca
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------------
# PCA component selection criteria
# ---------------------------------------------------------------------------

def malinowski_indicator(eigenvalue_matrix: np.ndarray) -> int:
    """Compute the Malinowski indicator to select the number of PCA components.

    Args:
        eigenvalue_matrix: Diagonal matrix of eigenvalues from the covariance matrix.

    Returns:
        Optimal number of principal components to keep.
    """
    num_components = eigenvalue_matrix.shape[0]
    indicator_values = np.full(num_components - 1, np.nan)

    for idx in range(num_components - 1):
        diagonal_sum = np.sum(
            np.diag(eigenvalue_matrix[idx:, idx:]) / (num_components - idx)
        )
        indicator_values[idx] = np.sqrt(diagonal_sum) / (num_components - idx) ** 2

    return int(np.argmin(indicator_values) + 1)


def nelson_coefficient(eigenvalue_matrix: np.ndarray) -> int:
    """Compute the Nelson coefficient of determination for component selection.

    Fits a linear model to the eigenvalues; chooses the point where R² exceeds 0.80.

    Args:
        eigenvalue_matrix: Diagonal matrix of eigenvalues from the covariance matrix.

    Returns:
        Optimal number of principal components to keep.
    """
    num_components = eigenvalue_matrix.shape[0]
    r_squared_values = np.full(num_components - 2, np.nan)

    for idx in range(num_components - 2):
        x_values = np.arange(idx, num_components)
        x_matrix = np.vstack([np.ones(len(x_values)), x_values]).T
        y_values = np.diag(eigenvalue_matrix[idx:, idx:])

        coefficients, _, _, _ = np.linalg.lstsq(x_matrix, y_values, rcond=None)
        y_predicted = x_matrix @ coefficients
        r_squared_values[idx] = r2_score(y_values, y_predicted)

    if not np.any(r_squared_values > 0.80):
        raise ValueError(
            "Nelson criterion: no R² value exceeds 0.80. "
            "Cannot determine the number of components."
        )
    return int(np.argmax(r_squared_values > 0.80) + 1)


def median_noise_estimation(eigenvalue_matrix: np.ndarray) -> int:
    """Estimate number of components using median noise estimation.

    Args:
        eigenvalue_matrix: Diagonal matrix of eigenvalues from the covariance matrix.

    Returns:
        Optimal number of principal components to keep.
    """
    diagonal_values = np.sqrt(np.diag(eigenvalue_matrix))
    median_value = np.median(
        diagonal_values[diagonal_values < 2 * np.median(diagonal_values)]
    )
    optimal_components = np.where(diagonal_values >= 1.29 * median_value)[0][-1] + 1
    return int(optimal_components)


# ---------------------------------------------------------------------------
# PCA denoising
# ---------------------------------------------------------------------------

def apply_pca_denoising(
    img: np.ndarray,
    criterion: str,
    mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Denoise a 4D MRI image using adaptive PCA.

    Args:
        img: 4D array (x, y, z, offsets). Should be B0-corrected and normalised.
        criterion: One of 'malinowski', 'nelson', 'median', 'mle'.
        mask: 3D binary mask array.

    Returns:
        Tuple of (denoised image, number of components used).
    """
    if img.ndim != 4:
        raise ValueError("img must be a 4D array.")
    valid = ("malinowski", "nelson", "median", "mle")
    if criterion not in valid:
        raise ValueError(f"criterion must be one of {valid}.")

    nx, ny, nz, nw = img.shape
    img = img.copy()
    img[np.isnan(img)] = 0
    c = np.transpose(img, (3, 0, 1, 2)).reshape(nw, nx * ny * nz)

    pca_full = PCA(n_components=None, random_state=42)
    pca_full.fit(c)

    criterion_fn = {
        "malinowski": malinowski_indicator,
        "nelson": nelson_coefficient,
        "median": median_noise_estimation,
    }
    if criterion == "mle":
        n_components = "mle"
    else:
        n_components = criterion_fn[criterion](np.diag(pca_full.explained_variance_))

    pca = PCA(n_components=n_components, random_state=42)
    denoised = pca.inverse_transform(pca.fit_transform(c))
    denoised = np.transpose(denoised.reshape(nw, nx, ny, nz), (1, 2, 3, 0))

    # Apply mask
    if mask.ndim == img.ndim - 1:
        for t in range(nw):
            denoised[:, :, :, t] = np.where(mask == 0, 0, denoised[:, :, :, t])
    elif mask.shape == denoised.shape:
        denoised[mask == 0] = 0
    else:
        raise ValueError("Mask shape does not match the denoised image shape.")

    return denoised, n_components


def apply_mppca_denoising(
    img: np.ndarray,
    mask: np.ndarray,
    patch_radius: int = 2,
    return_sigma: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Denoise a 4D MRI image using Marchenko-Pastur PCA (MP-PCA).

    Args:
        img: 4D array (x, y, z, offsets).
        mask: 3D binary mask array.
        patch_radius: Radius of local patches for MP-PCA. Default 2.
        return_sigma: If True, return the estimated noise map (sigma).

    Returns:
        Tuple of (denoised image, sigma map). sigma map is None when
        *return_sigma* is False.
    """
    if img.ndim != 4:
        raise ValueError("img must be a 4D array.")
    denoised, sigma = mppca(img, mask=mask, patch_radius=patch_radius, return_sigma=True)
    if not return_sigma:
        sigma = None
    return denoised, sigma


# ---------------------------------------------------------------------------
# Non-local means denoising
# ---------------------------------------------------------------------------

def apply_nlm_denoising(
    img: np.ndarray,
    sigma_array: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Apply non-local means denoising to a 4D image.

    Args:
        img: 4D array (x, y, z, offsets).
        sigma_array: Array of sigma values, one per offset.
        mask: 3D binary mask array.

    Returns:
        Denoised 4D image.
    """
    denoised = np.zeros(img.shape)
    for t in range(img.shape[3]):
        denoised[:, :, :, t] = nlmeans(
            img[:, :, :, t], sigma_array[t], mask=mask, rician=True
        )
    return denoised


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_denoising(
    img: np.ndarray,
    method: str,
    criterion: str = "mle",
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply denoising to a MRI image.

    Args:
        img: 4D array (x, y, z, offsets).
        method: 'pca', 'mppca', 'nlm', or None (no denoising).
        criterion: PCA criterion (only used when method='pca').
        mask: Optional 3D binary mask.

    Returns:
        Tuple of (denoised image, sigma map). sigma map is only non-None for
        method='mppca'; for other methods it is None.
    """
    if mask is None:
        mask = np.ones(img.shape[:3])

    if method == "pca":
        denoised, _ = apply_pca_denoising(img, criterion, mask)
        return denoised, None

    elif method == "mppca":
        return apply_mppca_denoising(img, mask)

    elif method == "nlm":
        sigmas = estimate_sigma(img)
        denoised = apply_nlm_denoising(img, sigmas, mask)
        return denoised, None

    elif method is None:
        return img, None
    
    else: 
        raise ValueError(f"Denoising method '{method}' not supported.")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def calculate_snr(original: np.ndarray, noisy: np.ndarray) -> float:
    """Calculate signal-to-noise ratio between two images (dB).

    Args:
        original: Reference image array.
        noisy: Noisy (or denoised) image array.

    Returns:
        SNR in decibels.
    """
    signal_power = np.var(original)
    noise_power = np.var(original - noisy)
    return float(10 * np.log10(signal_power / noise_power))
