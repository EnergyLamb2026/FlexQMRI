"""Functions to fit MRI data using least squares fitting."""

import warnings
from scipy.optimize import curve_fit, OptimizeWarning
import numpy as np
from torch.utils.data import DataLoader
from joblib import Parallel, delayed

from flexqmri.dataset import get_modality_and_data_type
from flexqmri.evaluation import utils
from flexqmri.utils.biophysical_model import get_model_from_config

# Suppress OptimizeWarning globally
warnings.filterwarnings('ignore', category=OptimizeWarning)


def _is_within_bounds(params: np.ndarray, fit_ranges: list) -> bool:
    """Check if all parameters are within their specified ranges.

    Args:
        params (np.ndarray): Parameter array to validate.
        fit_ranges (list): List of [min, max] ranges for each parameter.

    Returns:
        bool: True if all parameters lie within *fit_ranges*.
    """
    return all(fit_ranges[k][0] <= params[k] <= fit_ranges[k][1] for k in range(len(params)))


def _fit_single_voxel(
    i: int,
    signals: np.ndarray,
    x: np.ndarray,
    fit_ranges: list,
    model_fn,
    fitting_method: str,
    f_tol: float,
    x_tol: float,
    max_iter: int,
    n_fit: int,
) -> tuple:
    """Fit a single voxel using the specified fitting method.

    Removes NaN entries, performs up to ``max_retry`` rounds of ``n_fit``
    random restarts, and keeps the fit with the smallest residual that lies
    within *fit_ranges*. If no valid fit is found, the best result is clamped.

    Args:
        i (int): Voxel row index into *signals*.
        signals (np.ndarray): Signal array of shape ``(n_voxels, n_measurements)``.
        x (np.ndarray): Independent variable (e.g. b-values).
        fit_ranges (list): List of ``[min, max]`` ranges for each parameter.
        model_fn: Callable physical model compatible with ``scipy.optimize.curve_fit``.
        fitting_method (str): ``'lm'`` or ``'trf'``.
        f_tol (float): Tolerance for termination by cost-function change.
        x_tol (float): Tolerance for termination by variable change.
        max_iter (int): Maximum number of function evaluations.
        n_fit (int): Number of random-restart fits per retry round.

    Returns:
        tuple: ``(fitted_params, oob_count)`` where *fitted_params* is a 1-D
            ``np.ndarray`` in the original parameter space and *oob_count* is
            the number of out-of-bounds fits encountered.
    """
    signal = signals[i]
    x_voxel = x[i] if x.ndim > 1 else x
    n_params = len(fit_ranges)

    mask = ~np.isnan(signal).astype(bool)
    signal_clean = signal[mask]
    x_clean = x_voxel[mask]

    default_params = np.array([(r[0] + r[1]) / 2.0 for r in fit_ranges])
    max_retry = 5
    oob_count = 0

    for _ in range(max_retry):
        multi_popt = np.tile(default_params, (n_fit, 1))
        multi_infodict = np.zeros((n_fit, signal_clean.shape[0]))
        valid_fits = []

        for j in range(n_fit):
            try:
                start_points = tuple(np.random.uniform(r[0], r[1]) for r in fit_ranges)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OptimizeWarning)
                    if fitting_method == "lm":
                        multi_popt[j, :], _, infodict, _, _ = curve_fit(
                            model_fn,
                            x_clean,
                            signal_clean,
                            p0=start_points,
                            method=fitting_method,
                            ftol=f_tol,
                            xtol=x_tol,
                            maxfev=max_iter,
                            full_output=True,
                        )
                        multi_infodict[j, :] = infodict['fvec']
                    elif fitting_method == "trf":
                        bounds = ([r[0] for r in fit_ranges], [r[1] for r in fit_ranges])
                        multi_popt[j, :], _, infodict, _, _ = curve_fit(
                            model_fn,
                            x_clean,
                            signal_clean,
                            p0=start_points,
                            bounds=bounds,
                            method=fitting_method,
                            ftol=f_tol,
                            xtol=x_tol,
                            maxfev=max_iter,
                            full_output=True,
                        )
                        multi_infodict[j, :] = infodict['fvec']
                    else:
                        raise ValueError("Unknown or not supported fitting method")

                if _is_within_bounds(multi_popt[j, :], fit_ranges):
                    valid_fits.append(j)
                else:
                    oob_count += 1
            except RuntimeError:
                continue

        if valid_fits:
            best_valid_idx = min(valid_fits, key=lambda j: np.mean(np.abs(multi_infodict[j, :])))
            return multi_popt[best_valid_idx, :], oob_count

    best_fit_idx = np.argmin(np.mean(np.abs(multi_infodict), axis=1))
    best_params = multi_popt[best_fit_idx, :].copy()
    for k in range(n_params):
        best_params[k] = np.clip(best_params[k], fit_ranges[k][0], fit_ranges[k][1])

    return best_params, oob_count


def fit_loader(
    loader: DataLoader,
    phys_model_function,
    param_ranges: list,
    fitting_method: str = "trf",
    n_fit: int = 1,
    f_tol: float = 1e-6,
    x_tol: float = 1e-6,
    max_iter: int = 2000,
    data_type: str = None,
    modality: str = None,
):
    """Fit the LSQ model on the given data loader.

    Args:
        loader (DataLoader): DataLoader containing the data to fit.
        phys_model: Physical model function for curve fitting (e.g., ivim_model_function).
        param_ranges (list): List of [min, max] ranges for each parameter.
        fitting_method (str): Fitting method - 'lm' (Levenberg-Marquardt) or 'trf' (Trust Region Reflective).
        n_fit (int): Number of fits to perform per voxel (best result is kept).
        f_tol (float): Tolerance for termination by the change of the cost function.
        x_tol (float): Tolerance for termination by the change of the independent variables.
        max_iter (int): Maximum number of function evaluations.
        data_type (str): Type of data - 'synthetic'.
        modality (str): Modality of the data.

    Returns:
        results (dict): Dictionary containing:
            - For synthetic data: absolute errors for each parameter and corresponding b-values and noise
    """
    fit_ranges = param_ranges
    model_fn = phys_model_function

    results = utils.init_results_dict(modality)
    total_voxels = 0
    out_of_bounds_count = 0
    print(f"Starting fitting for modality: {modality}, data type: {data_type}")

    for test_batch in loader:
        signals, x, y_true_params, noise = utils.get_batch_data(
            test_batch, param_ranges
        )
        n_rows = signals.shape[0]
        total_voxels += n_rows

        batch_results = Parallel(n_jobs=-1)(
            delayed(_fit_single_voxel)(
                i, signals, x, fit_ranges, model_fn, fitting_method,
                f_tol, x_tol, max_iter, n_fit,
            )
            for i in range(n_rows)
        )
        y_pred_rows, oob_counts = zip(*batch_results)
        out_of_bounds_count += sum(oob_counts)
        y_pred = np.vstack(y_pred_rows)

        results = utils.get_batch_results(
            results, y_true_params, y_pred, noise, x, modality, param_ranges
        )

    print(f"Total voxels: {total_voxels}, out-of-bounds fits: {out_of_bounds_count}")
    return results


def fit_loader_from_config(loader: DataLoader, config: dict):
    """Convenience wrapper for fit_loader that extracts parameters from a config dict.

    This is useful when running from FlexQMRI scripts that use config files.

    Args:
        loader (DataLoader): DataLoader containing the data to fit.
        config (dict): Configuration dictionary with 'data', 'train', and 'fit' sections.

    Returns:
        results (dict): Fitting results (see fit_loader for details).
    """
    # Determine data type
    modality, data_type = get_modality_and_data_type(config)
    model = get_model_from_config(config)

    return fit_loader(
        loader=loader,
        phys_model_function=model.forward_scipy,
        param_ranges=model.param_ranges,
        fitting_method=config["train"]["model"],
        n_fit=config["fit"]["n_fit"],
        f_tol=config["fit"]["f_tol"],
        x_tol=config["fit"]["x_tol"],
        max_iter=config["train"]["max_iter"],
        data_type=data_type,
        modality=modality,
    )

