'''Utils functions for evaluation.'''

import os
from pathlib import Path

import torch 
import numpy as np 
import scipy.stats
from flexqmri.utils import biophysical_model

def get_modality_parameters(modality: str) -> list[str]: 
    '''Get the parameters corresponding to the modality.
    
    Args:
        modality (str): Modality of the data
    
    Returns:
        list[str]: List of parameter names corresponding to the modality

    Raises:
        ValueError: If modality is unknown or not supported.
    '''
    if modality == 'ivim':
        parameters = ['f', 'D', 'D*']
    elif modality == 't2star':
        parameters = ['t2star']
    else: 
        raise ValueError('Unknown or not supported modality')

    return parameters

def get_all_modality_params(modality: str) -> list[str]:
    """Get all fitted parameters for a modality, including nuisance parameters like S0.

    Unlike get_modality_parameters, this includes every column produced by the biophysical
    model (e.g. S0), matching the column order of y_pred / y_true arrays.

    Args:
        modality (str): Modality of the data.

    Returns:
        list[str]: Parameter names in column order.

    Raises:
        ValueError: If modality is unknown or not supported.

    Notes:
        Add new modalities here and in get_modality_parameters; no other code needs changing.
    """
    params = {
        'ivim': ["S0", "f", "D", "D*"],
        't2star': ["S0", "t2star"],
    }
    if modality not in params:
        raise ValueError(f'Unknown or not supported modality: {modality}')
    return params[modality]


def init_results_dict(modality: str) -> dict:
    """Initialize results dictionary based on modality.

    Args:
        modality (str): Modality of the data ('ivim' or 't2star').

    Returns:
        dict: Initialized results dictionary.

    Raises:
        ValueError: If modality is unknown or not supported.
    """
    params = get_all_modality_params(modality)
    results = {key: [] for key in [f"{p}_re" for p in params] + [f"{p}_mse" for p in params] + ["noise", "x", "predictions", "y_true"]}
    return results


def accumulate_results(target: dict, source: dict) -> dict:
    """Extend each list field of ``target`` in place with the matching field of ``source``.

    Used to aggregate per-length results into a single results dict (e.g. across the
    fixed-length models of a multi-fixed run).

    Args:
        target (dict): Accumulator results dict (from init_results_dict).
        source (dict): A single run's results dict to merge in.

    Returns:
        dict: The same ``target`` dict, with each list field extended.
    """
    for key in target:
        if key in source:
            target[key].extend(source[key])
    return target

def compute_bootstrap_ci(data: np.ndarray, seed: int = 42, confidence: float = 0.95, n_resamples: int = 10000, statistic=None) -> dict:
    """Compute bootstrap confidence interval using the percentile method.

    Args:
        data (np.ndarray): 1D array of values to compute CI on.
        seed (int): Seed for the numpy random Generator (default: 42).
        confidence (float): Confidence level (default: 0.95 for 95% CI).
        n_resamples (int): Number of bootstrap resamples (default: 10000).
        statistic (callable, optional): Aggregation function applied to each resample.
            Defaults to np.mean. Use ``lambda x: np.sqrt(np.mean(x))`` for NRMSE
            when ``data`` contains squared normalized errors.

    Returns:
        dict: Dictionary with 'mean', 'ci_lower', 'ci_upper', 'std_error'.
    """
    if statistic is None:
        statistic = np.mean
    rng = np.random.default_rng(seed)
    result = scipy.stats.bootstrap(
        (data,),
        statistic,
        confidence_level=confidence,
        method='percentile',
        random_state=rng,
        n_resamples=n_resamples
    )
    return {
        'mean': float(statistic(data)),
        'ci_lower': float(result.confidence_interval.low),
        'ci_upper': float(result.confidence_interval.high),
        'std_error': float(result.standard_error)
    }


def aggregate_results_across_seeds(all_results: list, seed_nbr: int, seed: int = 42, confidence: float = 0.95) -> dict:
    """Aggregate results from multiple seed runs and compute confidence intervals using scipy bootstrap with the percentile method.

    Reference: https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html#method-4-confidence-intervals-from-retraining-models-with-different-random-seeds

    Args:
        all_results (list): List of result dictionaries from each seed run
        seed_nbr (int): Number of seeds used (must be > 1)
        seed (int): Seed for the numpy random Generator used in bootstrap (default: 42)
        confidence (float): Confidence level (default: 0.95 for 95% CI)

    Returns:
        dict: Aggregated results with means and confidence intervals for each parameter

    Raises:
        ValueError: If seed_nbr is 1. Use the raw results dict directly for a single seed.
    """
    if seed_nbr == 1:
        raise ValueError("aggregate_results_across_seeds requires seed_nbr > 1. Use the raw results dict directly for a single seed.")
    
    # Determine parameters from first result (exclude 'noise' and 'x')
    first_result = all_results[0]
    parameters = [key for key in first_result.keys() if key not in ['noise', 'x', 'predictions']]
    parameters = sorted(parameters)  # Sort for consistent ordering
    
    aggregated = {}
    
    print("\n" + "="*80)
    print(f"RESULTS AGGREGATED ACROSS {seed_nbr} SEEDS")
    print("="*80)
    
    for param in parameters:
        # Collect errors for this parameter across all seeds
        param_errors = []
        for result in all_results:
            param_errors.append(result[param])
        
        # Convert to numpy array, handling both lists and tensors
        param_errors_array_list = []
        for errors_list in param_errors:
            numpy_array = torch.stack(errors_list).cpu().numpy()
            param_errors_array_list.append(numpy_array)
        
        # Stack all errors: Shape (seed_nbr, n_samples)
        param_errors_array = np.stack(param_errors_array_list)

        # Compute mean error per seed: Shape (seed_nbr,)
        mean_errors_per_seed = np.mean(param_errors_array, axis=1)

        # Compute bootstrap CI on the per-seed mean errors
        aggregated[param] = compute_bootstrap_ci(
            mean_errors_per_seed, seed=seed, confidence=confidence
        )
        
        ci = aggregated[param]
        print(f"\n{param} Parameter:")
        print(f"  Mean Error: {ci['mean']:.9f}")
        print(f"  {confidence*100:.0f}% CI (bootstrap percentile): [{ci['ci_lower']:.9f}, {ci['ci_upper']:.9f}]")
        print(f"  Std Error: {ci['std_error']:.9f}")
    
    print("\n" + "="*80)
    return aggregated



def get_batch_data(batch_data, param_ranges: list) -> tuple: 
    '''Get the data from a batch for LSQ fitting (no model predictions).

    Args:
        batch_data: Batch of data from the DataLoader
        param_ranges (list): List of [min, max] ranges for each parameter.
    Returns:
        tuple: Contains the following:
            - signals_noisy (np.ndarray): Signals from the input data
            - x (np.ndarray): b-values from the input data
            - y_true_params (np.ndarray): True parameter values
            - noise_test (np.ndarray): Noise from the input data
    '''
    # get data from batch
    x_noisy, y_true_coeffs, noise_test = batch_data
    x = x_noisy[:, :, 0].cpu().numpy()
    signals_noisy = x_noisy[:, :, 1].cpu().numpy()

    y_true_params = biophysical_model.rescale_coeffs_torch(param_ranges, y_true_coeffs).cpu().numpy()
    noise_test = noise_test.cpu().numpy()

    return signals_noisy, x, y_true_params, noise_test

def get_net_outputs(config: dict, batch_data, model: torch.nn.Module, device='cpu', atol=1e-5, rtol=1e-3) -> tuple: 
    '''Get the predicted and true parameter values from the model for a given batch of data.    

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        batch_data: Batch of data from the DataLoader
        model (torch.nn.Module): Trained NeuralCDE or MLP model
        device (str): Device to use for the tensors ('cuda' or 'cpu')
        atol (float): Absolute tolerance for the adaptive ODE solver (default: 1e-5)
        rtol (float): Relative tolerance for the adaptive ODE solver (default: 1e-3)

    Returns:
        tuple: Contains the following:
            - y_pred_coeffs (torch.Tensor): Predicted coefficient values
            - y_true_coeffs (torch.Tensor): True coefficient values
            - y_pred_params (torch.Tensor): Predicted parameter values
            - y_true_params (torch.Tensor): True parameter values
            - x (torch.Tensor): measurement offsets from the input data
            - noise (torch.Tensor): Noise from the input data

    Raises:
        ValueError: If the model type in config is unknown or not implemented.
    '''

    if config["train"]["model"] == 'ncde': 
        if config["train"].get("interpolation_during_training", False):
            x_noisy, y_true_coeffs, noise = batch_data
            x_noisy, y_true_coeffs, noise = x_noisy.to(device), y_true_coeffs.to(device), noise.to(device)
            x = x_noisy[:, :, 0] 
            y_pred = model(x_noisy, atol=atol, rtol=rtol)
        else:
            coeffs, x_noisy, y_true_coeffs, noise = batch_data
            coeffs, x_noisy, y_true_coeffs, noise = coeffs.to(device), x_noisy.to(device), y_true_coeffs.to(device), noise.to(device)
            x = x_noisy[:, :, 0] 
            y_pred = model(coeffs, atol=atol, rtol=rtol)
    elif config["train"]["model"] == 'mlp':
        x_noisy, y_true_coeffs, noise = batch_data
        x_noisy, y_true_coeffs, noise = x_noisy.to(device), y_true_coeffs.to(device), noise.to(device)
        x = x_noisy[:, :, 0]
        y_pred = model(x_noisy)
    elif config["train"]["model"] == 'transformer':
        x_noisy, y_true_coeffs, noise = batch_data
        x_noisy, y_true_coeffs, noise = x_noisy.to(device), y_true_coeffs.to(device), noise.to(device)
        x = x_noisy[:, :, 0]
        y_pred = model(x_noisy)
    else:
        raise ValueError(f"Unknown/non implemented model type: {config['train']['model']}")

    if config["train"]["output_activation"] == 'sigmoid':
        y_pred_coeffs = torch.sigmoid(y_pred)
    else: # clamp between 0 and 1
        y_pred_coeffs = torch.clamp(y_pred, 0.0, 1.0)

    _model = biophysical_model.get_model_from_config(config)
    y_pred_params = _model.rescale_coeffs_torch(y_pred_coeffs)
    y_true_params = _model.rescale_coeffs_torch(y_true_coeffs)

    return y_pred_coeffs, y_true_coeffs, y_pred_params, y_true_params, x, noise

def get_batch_results(results: dict, y_true, y_pred, noise_test, x_test, modality: str, param_ranges: list) -> dict:
    '''Get the fitting results for a batch of data.

    Args:
        results (dict): Dictionary to store the results
        y_true (torch.Tensor or None): True parameter values. Pass None when ground truth is unavailable
            (e.g. real data); error keys are skipped and y_pred is stored under 'predictions'.
        y_pred (torch.Tensor or np.ndarray): Predicted parameter values.
        noise_test (torch.Tensor or None): Noise data from the test set. Pass None when unavailable.
        x_test (torch.Tensor or np.ndarray): B-values / TEs from the test set.
        modality (str): Modality of the data (default is 'ivim')
        param_ranges (list): List of ``[min, max]`` per parameter (column order must match
            ``y_pred``/``y_true``). Used to compute range-normalized error.

    Returns:
        dict: Updated dictionary containing squared normalized errors
            ({param}_re = ((pred-true)/(max-min))²; sqrt(mean) gives NRMSE),
            MSE ({param}_mse), noise, and x when y_true is provided; or predictions and x
            when y_true is None.

    Raises:
        ValueError: If modality is unknown or not supported.

    Notes:
        Per-parameter squared normalized error is stored so that NRMSE =
        sqrt(mean(_re)) at aggregation time. For per-sample absolute errors
        (e.g. boxplots), apply sqrt element-wise: sqrt(_re) = |pred-true|/(max-min).
    '''

    # Deal with Torch/Numpy formats and ensure all data is on CPU for consistent processing
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    if isinstance(x_test, np.ndarray):
        x_test = torch.tensor(x_test)
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(noise_test, np.ndarray):
        noise_test = torch.tensor(noise_test)

    # Store predictions and x values
    results['x'].extend(x_test)
    results['predictions'].extend(y_pred)
    if y_true is None:
        return results
    results['y_true'].extend(y_true)

    ranges = torch.tensor(
        [r[1] - r[0] for r in param_ranges], dtype=y_pred.dtype, device=y_pred.device
    )
    sq_norm_error = ((y_pred - y_true) / ranges) ** 2  # squared normalized error; sqrt(mean) = NRMSE
    mse = (y_pred - y_true) ** 2
    results['noise'].extend(noise_test)
    for i, p in enumerate(get_all_modality_params(modality)):
        results[f'{p}_re'].extend(sq_norm_error[:, i])
        results[f'{p}_mse'].extend(mse[:, i])

    return results


def compute_results_bootstrap_ci(results: dict, modality: str, seed: int = 42, confidence: float = 0.95) -> dict:
    """Compute bootstrap confidence intervals on accumulated batch results.

    This should be called after all batches have been processed and their
    errors accumulated in ``results`` via :func:`get_batch_results`.

    Args:
        results (dict): Dictionary containing accumulated per-sample errors
            for each parameter (lists of tensors) plus 'noise' and 'x'.
        modality (str): Modality of the data ('ivim', 't2star', …).
        seed (int): Seed for the numpy random Generator used in bootstrap (default: 42).
        confidence (float): Confidence level (default: 0.95 for 95% CI).

    Returns:
        dict: Mapping from parameter name to a dict with keys
            'mean', 'ci_lower', 'ci_upper', 'std_error'.
    """
    parameters = get_modality_parameters(modality)
    nrmse_stat = lambda x: np.sqrt(np.mean(x))
    ci_results = {}
    for param in parameters:
        key = f'{param}_re'
        sq_errors = torch.stack(results[key]).cpu().numpy() if isinstance(results[key][0], torch.Tensor) else np.array(results[key])
        ci_results[param] = compute_bootstrap_ci(sq_errors, seed=seed, confidence=confidence, statistic=nrmse_stat)
    return ci_results


def print_fit_results(results: dict, modality: str, seed: int | None = None, confidence: float = 0.95) -> None:
    '''Print fit results with bootstrap confidence intervals.

    Args:
        results (dict): Dictionary containing the errors for each parameter and the corresponding x and noise
        modality (str): Modality of the data
        seed (int | None): Seed for the numpy random Generator used in bootstrap (default: None)
        confidence (float): Confidence level for bootstrap CI (default: 0.95)

    Returns:
        None
    '''
    parameters = get_modality_parameters(modality)

    # NRMSE: sqrt(mean(((pred - true) / range)^2))
    nrmse = {p: float(np.sqrt(np.mean(torch.stack(results[f'{p}_re']).cpu().numpy()))) for p in parameters}
    print('Test NRMSE:                 ' + ', '.join(f'{p}: {nrmse[p]:.3e}' for p in parameters))
    ci_results = compute_results_bootstrap_ci(results, modality, seed=seed, confidence=confidence)
    for param in parameters:
        ci = ci_results[param]
        print(f"  {param} bootstrap {confidence*100:.0f}% CI (percentile): "
              f"[{ci['ci_lower']:.3e}, {ci['ci_upper']:.3e}]")
        
    # Mean squared error
    mean_mse = {p: torch.mean(torch.tensor(results[f'{p}_mse'])).item() for p in parameters}
    print('Test Mean MSE:            ' + ', '.join(f'{p}: {mean_mse[p]:.3e}' for p in parameters))
    for param in parameters:
        mse_vals = results[f'{param}_mse']
        errors = torch.stack(mse_vals).cpu().numpy() if isinstance(mse_vals[0], torch.Tensor) else np.array(mse_vals)
        ci = compute_bootstrap_ci(errors, seed=seed, confidence=confidence)
        print(f"  {param} MSE bootstrap {confidence*100:.0f}% CI (percentile): "
              f"[{ci['ci_lower']:.3e}, {ci['ci_upper']:.3e}]")

    # Spearman ρ with bootstrap CI (available when results contains 'spearman_r')
    spearman = results.get('spearman_r')
    if spearman is not None:
        pred_vs_true = spearman.get('pred_vs_true', {})
        pred_vs_true_ci = spearman.get('pred_vs_true_ci', {})
        print('Spearman ρ (pred vs true): ' + ', '.join(
            f'{p}: {pred_vs_true[p]:.4f}' for p in parameters if p in pred_vs_true
        ))
        for param in parameters:
            if param in pred_vs_true_ci:
                ci = pred_vs_true_ci[param]
                print(f"  {param} ρ bootstrap {confidence*100:.0f}% CI (percentile): "
                      f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")


def compute_spearman_r(
    results: dict,
    modality: str,
    seed: int = 42,
    confidence: float = 0.95,
    n_resamples: int = 10000,
) -> dict:
    """Compute Spearman ρ with bootstrap CI per parameter, and cross-parameter correlation matrix.

    Spearman rank correlation is used instead of Pearson because it measures whether
    the model correctly ranks voxels by their parameter value (monotonic relationship),
    without assuming linearity, and is robust to the outliers common in MRI fitting.

    Bootstrap CI is computed by resampling (pred, true) pairs jointly (paired=True),
    recomputing ρ on each resample, and taking percentile bounds.

    Args:
        results (dict): Results dict containing 'predictions' and 'y_true' lists of tensors.
        modality (str): Modality of the data ('ivim', 't2star', …).
        seed (int): Seed for the numpy random Generator used in bootstrap (default: 42).
        confidence (float): Confidence level for the bootstrap CI (default: 0.95).
        n_resamples (int): Number of bootstrap resamples (default: 10000).

    Returns:
        dict: Keys:
            'pred_vs_true' (dict): Maps each parameter name to its Spearman ρ.
            'pred_vs_true_ci' (dict): Maps each parameter name to a CI dict with keys
                'ci_lower', 'ci_upper', 'std_error'.
            'cross_param' (np.ndarray): Spearman correlation matrix of shape (n_params, n_params)
                among predicted biophysical parameters.
            'param_names' (list): Parameter names in column order.

    Notes:
        S0 is excluded; only parameters from get_modality_parameters are used.
    """
    all_param_names = get_all_modality_params(modality)
    bio_params = get_modality_parameters(modality)
    bio_indices = [all_param_names.index(p) for p in bio_params]

    y_pred = torch.stack(results['predictions']).cpu().numpy()  # (N, n_all_params)
    y_true = torch.stack(results['y_true']).cpu().numpy()       # (N, n_all_params)

    rng = np.random.default_rng(seed)
    pred_vs_true = {}
    pred_vs_true_ci = {}

    for param, idx in zip(bio_params, bio_indices):
        p, t = y_pred[:, idx], y_true[:, idx]
        rho, _ = scipy.stats.spearmanr(p, t)
        pred_vs_true[param] = float(rho)

        boot = scipy.stats.bootstrap(
            (p, t),
            statistic=lambda a, b: scipy.stats.spearmanr(a, b).statistic,
            paired=True,
            n_resamples=n_resamples,
            confidence_level=confidence,
            method='percentile',
            random_state=rng,
        )
        pred_vs_true_ci[param] = {
            'ci_lower': float(boot.confidence_interval.low),
            'ci_upper': float(boot.confidence_interval.high),
            'std_error': float(boot.standard_error),
        }

    pred_bio = y_pred[:, bio_indices]
    cross_param = scipy.stats.spearmanr(pred_bio).statistic if len(bio_params) > 1 else None

    return {
        'pred_vs_true': pred_vs_true,
        'pred_vs_true_ci': pred_vs_true_ci,
        'cross_param': cross_param,
        'param_names': bio_params,
    }


def test_network(config: dict, test_loader: torch.utils.data.DataLoader, model: torch.nn.Module, modality: str, device='cpu', atol=1e-5, rtol=1e-3) -> dict:
    """Make predictions for a given model and compute errors against ground truth.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        test_loader (torch.utils.data.DataLoader): Test dataset loader
        model (torch.nn.Module): Trained NeuralCDE or MLP model
        modality (str): Modality of the data ('ivim', 't2star', etc.)
        device (str): Device to use for the tensors ('cuda' or 'cpu')
        atol (float): Absolute tolerance for the adaptive ODE solver
        rtol (float): Relative tolerance for the adaptive ODE solver

    Returns:
        dict: Dictionary containing the errors for each parameter and the corresponding measurement values and noise
    """
    results = init_results_dict(modality)
    param_ranges = biophysical_model.get_model_from_config(config).param_ranges

    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_data in test_loader:
            _, _, y_pred_params, y_true_params, x_test, noise_test = get_net_outputs(config, batch_data, model, device, atol=atol, rtol=rtol)
            results = get_batch_results(results, y_true_params, y_pred_params, noise_test, x_test, modality, param_ranges)

    return results


# ---------------------------------------------------------------------------
# Comparison utilities  (used by scripts/compare_experiments.py)
# ---------------------------------------------------------------------------

def get_computed_metrics(global_run_id: str, results_root: str = 'results', modality: str = None) -> list:
    """Find all metrics.pt files under results_root whose path contains global_run_id.

    Args:
        global_run_id (str): The run ID assigned to one experiment.
        results_root (str): Root directory to search (default: 'results').
        modality (str, optional): If given, restrict the search to the
            ``{modality}/`` subtree, so a run ID shared across modalities is not
            matched in the wrong one. Defaults to None (search all modalities).

    Returns:
        list[Path]: Sorted list of matching metrics.pt paths.

    Raises:
        FileNotFoundError: If no metrics.pt is found for this global_run_id.
    """
    root = Path(results_root)
    prefix = f'{modality}/' if modality else ''
    matches = sorted(p for p in root.glob(f'{prefix}**/{global_run_id}/**/metrics.pt'))
    if not matches:
        raise FileNotFoundError(
            f"No metrics.pt found for global_run_id='{global_run_id}' under {results_root}"
        )
    return matches


def load_and_merge_metrics(metrics_files: list) -> dict:
    """Load and merge per-sample errors from multiple metrics.pt files.

    Used to aggregate results across multiple seed runs that share a global_run_id.

    Args:
        metrics_files (list[Path]): List of metrics.pt paths to merge.

    Returns:
        dict: Merged results with per-sample error lists concatenated across files.
    """
    merged = None
    for f in metrics_files:
        result = torch.load(f, weights_only=False)
        if merged is None:
            merged = {
                k: list(v) if isinstance(v, (list, torch.Tensor)) else v
                for k, v in result.items()
            }
        else:
            for k in merged:
                if k in result and isinstance(merged[k], list):
                    v = result[k]
                    merged[k].extend(v if isinstance(v, list) else list(v))
    return merged


def shorten_labels(run_ids: list) -> list:
    """Strip the longest common prefix from a list of run IDs for display.

    Args:
        run_ids (list[str]): List of global_run_ids.

    Returns:
        list[str]: Shortened labels with common prefix removed.
    """
    if len(run_ids) <= 1:
        return list(run_ids)
    prefix = os.path.commonprefix(run_ids)
    last_sep = prefix.rfind('_')
    prefix = prefix[:last_sep + 1] if last_sep >= 0 else prefix
    return [rid[len(prefix):] or rid for rid in run_ids]


def has_mse_keys(all_experiments: dict, modality: str) -> bool:
    """Check whether MSE keys are present in all loaded experiment results.

    Args:
        all_experiments (dict): Mapping label -> results dict.
        modality (str): Modality string.

    Returns:
        bool: True if all experiments contain MSE keys for all parameters.
    """
    parameters = get_modality_parameters(modality)
    return all(
        f'{p}_mse' in results
        for results in all_experiments.values()
        for p in parameters
    )