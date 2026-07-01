"""Summary table generation for evaluation."""

import numpy as np
import pandas as pd
import torch

from flexqmri.evaluation.utils import (
    get_modality_parameters,
    compute_bootstrap_ci,
    shorten_labels,
)

def build_summary_table_by_noise(
    all_experiments: dict, modality: str, metric: str = 'nrmse', n_bins: int = 5
) -> pd.DataFrame:
    """Build a summary DataFrame with mean and bootstrap CI per parameter, grouped by SNR level.

    One row per (experiment, SNR) pair. When SNR is continuous (more unique values than
    ``n_bins``), samples are divided into ``n_bins`` equal-frequency quantile bins and the
    bin midpoint is used as the SNR label.

    Args:
        all_experiments (dict): Mapping label -> results dict.
        modality (str): Modality string (e.g. 'ivim').
        metric (str): 'nrmse' for NRMSE, 'mse' for squared error.
        n_bins (int): Number of quantile bins when SNR is continuous. Defaults to 5.

    Returns:
        pd.DataFrame: Rows = (label, SNR), columns include 'experiment', 'SNR', one
            display column per physical parameter, plus raw mean/CI columns.
    """
    parameters = get_modality_parameters(modality)
    short = shorten_labels(list(all_experiments.keys()))
    fmt = '{:.2e}'
    rows = []
    for short_label, (_, results) in zip(short, all_experiments.items()):
        noise_data = torch.stack(results['noise']).cpu().numpy().flatten()
        unique_snr = np.unique(noise_data)
        if len(unique_snr) > n_bins:
            bins_series = pd.qcut(pd.Series(noise_data), q=n_bins, duplicates='drop')
            groups = [
                (round(cat.mid, 1), (bins_series == cat).values)
                for cat in bins_series.cat.categories
            ]
        else:
            groups = [(snr, noise_data == snr) for snr in sorted(unique_snr)]

        col_suffix = 'nrmse' if metric == 'nrmse' else 'mse'
        for snr_label, mask in groups:
            row = {'experiment': short_label, 'SNR': snr_label}
            for param in parameters:
                key = f'{param}_re' if metric == 'nrmse' else f'{param}_mse'
                errors = results[key]
                errors_np = (
                    torch.stack(errors).cpu().numpy()
                    if isinstance(errors[0], torch.Tensor)
                    else np.array(errors)
                )
                stat = (lambda x: np.sqrt(np.mean(x))) if metric == 'nrmse' else np.mean
                ci = compute_bootstrap_ci(errors_np[mask], statistic=stat)
                row[f'{param}_mean'] = ci['mean']
                row[f'{param}_ci_lower'] = ci['ci_lower']
                row[f'{param}_ci_upper'] = ci['ci_upper']
                row[f'{param}_{col_suffix}'] = (
                    f"{fmt.format(ci['mean'])} [{fmt.format(ci['ci_lower'])}, {fmt.format(ci['ci_upper'])}]"
                )
            rows.append(row)
    return pd.DataFrame(rows)

def build_summary_table_by_sampling(
    all_experiments: dict, modality: str, metric: str = 'nrmse'
) -> pd.DataFrame:
    """Build a summary DataFrame with mean and bootstrap CI per parameter, grouped by number of measurements.

    One row per (experiment, n_measurements) pair. Measurement count is inferred from the
    number of non-NaN entries in each sample's 'x' tensor (b-values or TEs).

    Args:
        all_experiments (dict): Mapping label -> results dict.
        modality (str): Modality string (e.g. 'ivim').
        metric (str): 'nrmse' for NRMSE, 'mse' for squared error.

    Returns:
        pd.DataFrame: Rows = (label, n_measurements), columns include 'experiment',
            'n_measurements', one display column per physical parameter, plus raw mean/CI columns.
    """
    parameters = get_modality_parameters(modality)
    short = shorten_labels(list(all_experiments.keys()))
    fmt = '{:.2e}'
    rows = []
    for short_label, (_, results) in zip(short, all_experiments.items()):
        x_list = results['x']
        try:
            x_data = torch.stack(x_list).cpu().numpy()
        except RuntimeError:
            padded = torch.nn.utils.rnn.pad_sequence(x_list, batch_first=True, padding_value=np.nan)
            x_data = padded.cpu().numpy()
        n_meas = np.sum(~np.isnan(x_data), axis=1)
        col_suffix = 'nrmse' if metric == 'nrmse' else 'mse'
        for n in sorted(np.unique(n_meas)):
            mask = n_meas == n
            row = {'experiment': short_label, 'n_measurements': int(n)}
            for param in parameters:
                key = f'{param}_re' if metric == 'nrmse' else f'{param}_mse'
                errors = results[key]
                errors_np = (
                    torch.stack(errors).cpu().numpy()
                    if isinstance(errors[0], torch.Tensor)
                    else np.array(errors)
                )
                stat = (lambda x: np.sqrt(np.mean(x))) if metric == 'nrmse' else np.mean
                ci = compute_bootstrap_ci(errors_np[mask], statistic=stat)
                row[f'{param}_mean'] = ci['mean']
                row[f'{param}_ci_lower'] = ci['ci_lower']
                row[f'{param}_ci_upper'] = ci['ci_upper']
                row[f'{param}_{col_suffix}'] = (
                    f"{fmt.format(ci['mean'])} [{fmt.format(ci['ci_lower'])}, {fmt.format(ci['ci_upper'])}]"
                )
            rows.append(row)
    return pd.DataFrame(rows)

def build_summary_table(all_experiments: dict, modality: str, metric: str = 'nrmse') -> pd.DataFrame:
    """Build a summary DataFrame with mean and bootstrap CI per parameter.

    Args:
        all_experiments (dict): Mapping label -> results dict.
        modality (str): Modality string (e.g. 'ivim').
        metric (str): 'nrmse' for NRMSE, 'mse' for squared error.

    Returns:
        pd.DataFrame: Rows = labels, one display column per physical parameter.
    """
    parameters = get_modality_parameters(modality)
    short = shorten_labels(list(all_experiments.keys()))
    fmt = '{:.2e}'
    col_suffix = 'nrmse' if metric == 'nrmse' else 'mse'
    rows = []
    for short_label, (label, results) in zip(short, all_experiments.items()):
        row = {'experiment': short_label}
        t = results.get('training_time_seconds')
        if t is not None:
            h, rem = divmod(int(t), 3600)
            m, s = divmod(rem, 60)
            row['training_time'] = f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")
        else:
            row['training_time'] = 'N/A'
        for param in parameters:
            key = f'{param}_re' if metric == 'nrmse' else f'{param}_mse'
            errors = results[key]
            errors_np = (
                torch.stack(errors).cpu().numpy()
                if isinstance(errors[0], torch.Tensor)
                else np.array(errors)
            )
            stat = (lambda x: np.sqrt(np.mean(x))) if metric == 'nrmse' else np.mean
            ci = compute_bootstrap_ci(errors_np, statistic=stat)
            row[f'{param}_mean'] = ci['mean']
            row[f'{param}_ci_lower'] = ci['ci_lower']
            row[f'{param}_ci_upper'] = ci['ci_upper']
            row[f'{param}_{col_suffix}'] = (
                f"{fmt.format(ci['mean'])} [{fmt.format(ci['ci_lower'])}, {fmt.format(ci['ci_upper'])}]"
            )
        rows.append(row)
    return pd.DataFrame(rows)
