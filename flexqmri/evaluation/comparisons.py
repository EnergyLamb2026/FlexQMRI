"""Comparison functions for evaluating multiple experiments."""

from pathlib import Path

import numpy as np
import torch

from flexqmri.evaluation.utils import (
    get_computed_metrics,
    get_modality_parameters,
    has_mse_keys,
    load_and_merge_metrics,
)
from flexqmri.evaluation.tables import (
    build_summary_table,
    build_summary_table_by_noise,
    build_summary_table_by_sampling,
)


def load_experiments(global_run_ids: list, results_root: str, modality: str = None) -> dict:
    """Load and merge metrics for each global run ID.

    Args:
        global_run_ids (list): List of global run IDs to load.
        results_root (str): Root directory to scan for metrics.pt files.
        modality (str, optional): If given, restrict discovery to the
            ``{modality}/`` subtree, so a run ID shared across modalities is not
            matched in the wrong one. Defaults to None (search all modalities).

    Returns:
        dict: Mapping from run ID to merged metrics dictionary.
    """
    from flexqmri.evaluation import recompute

    all_experiments = {}
    for run_id in global_run_ids:
        print(f"Discovering metrics for '{run_id}' in {results_root}/ ...")
        recompute.ensure_metrics(run_id, results_root, modality)
        metric_files = get_computed_metrics(run_id, results_root, modality)
        print(f"  Found {len(metric_files)} metric file(s): {[str(f) for f in metric_files]}")
        all_experiments[run_id] = load_and_merge_metrics(metric_files)
    return all_experiments


def _save_metric_summary(
    all_experiments: dict,
    modality: str,
    output_dir: Path,
    name: str,
    metric: str,
) -> None:
    """Build, print, and save the summary table (CSV) and boxplots for one metric type.

    Args:
        all_experiments (dict): Mapping from run ID to metrics dictionary.
        modality (str): Modality of the data.
        output_dir (Path): Directory to save outputs.
        name (str): Comparison name used as filename prefix.
        metric (str): Metric type, either 'nrmse' or 'mse'.
    """
    parameters = get_modality_parameters(modality)
    metric_suffix = 'nrmse' if metric == 'nrmse' else 'mse'
    display_cols = ['experiment', 'training_time'] + [f'{p}_{metric_suffix}' for p in parameters]
    df = build_summary_table(all_experiments, modality, metric=metric)
    print(f'\n{"="*80}')
    print(f'{metric_suffix.upper()} ERROR SUMMARY')
    print(f'{"="*80}')
    print(df[display_cols].to_string(index=False))
    csv_path = output_dir / f'{name}_{metric_suffix}_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n{metric_suffix.upper()} CSV saved to {csv_path}')


def has_noise_variation(all_experiments: dict) -> bool:
    """Return True if any experiment has more than one unique SNR level.

    Args:
        all_experiments (dict): Mapping from run ID to metrics dictionary.

    Returns:
        bool: True if at least one experiment contains multiple distinct SNR values.
    """
    for results in all_experiments.values():
        if not results.get('noise'):
            return False
        noise = torch.stack(results['noise']).cpu().numpy().flatten()
        if len(np.unique(noise)) > 1:
            return True
    return False


def has_sampling_variation(all_experiments: dict) -> bool:
    """Return True if any experiment has more than one unique measurement count.

    Args:
        all_experiments (dict): Mapping from run ID to metrics dictionary.

    Returns:
        bool: True if at least one experiment contains samples with different numbers of measurements.
    """
    for results in all_experiments.values():
        if not results.get('x'):
            return False
        x_list = results['x']
        try:
            x_data = torch.stack(x_list).cpu().numpy()
        except RuntimeError:
            padded = torch.nn.utils.rnn.pad_sequence(x_list, batch_first=True, padding_value=np.nan)
            x_data = padded.cpu().numpy()
        n_meas = np.sum(~np.isnan(x_data), axis=1)
        if len(np.unique(n_meas)) > 1:
            return True
    return False


def _save_grouped_metric_summary(
    all_experiments: dict,
    modality: str,
    output_dir: Path,
    name: str,
    metric: str,
    group_by: str,
) -> None:
    """Build, print, and save the grouped summary table (CSV) for one metric type.

    Args:
        all_experiments (dict): Mapping from run ID to metrics dictionary.
        modality (str): Modality of the data.
        output_dir (Path): Directory to save outputs.
        name (str): Comparison name used as filename prefix.
        metric (str): Metric type, either 'nrmse' or 'mse'.
        group_by (str): Grouping variable, either 'noise' (by SNR) or 'sampling' (by n_measurements).
    """
    parameters = get_modality_parameters(modality)
    metric_suffix = 'nrmse' if metric == 'nrmse' else 'mse'
    group_col = 'SNR' if group_by == 'noise' else 'n_measurements'
    group_label = 'snr' if group_by == 'noise' else 'sampling'
    display_cols = ['experiment', group_col] + [f'{p}_{metric_suffix}' for p in parameters]

    if group_by == 'noise':
        df = build_summary_table_by_noise(all_experiments, modality, metric=metric)
    else:
        df = build_summary_table_by_sampling(all_experiments, modality, metric=metric)

    print(f'\n{"="*80}')
    print(f'{metric_suffix.upper()} ERROR BY {group_col.upper()}')
    print(f'{"="*80}')
    print(df[display_cols].to_string(index=False))

    csv_path = output_dir / f'{name}_{metric_suffix}_by_{group_label}.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n{metric_suffix.upper()} by-{group_label} CSV saved to {csv_path}')


def run_comparison(all_experiments: dict, modality: str, output_dir: Path, name: str) -> None:
    """Save raw results, summary tables (CSV), and boxplots for all metrics.

    Args:
        all_experiments (dict): Mapping from run ID to metrics dictionary.
        modality (str): Modality of the data.
        output_dir (Path): Directory to save outputs.
        name (str): Comparison name used as filename prefix.
    """
    raw_path = output_dir / f'{name}_results.pt'
    torch.save(all_experiments, raw_path)
    print(f'\nRaw results saved to {raw_path}')

    _save_metric_summary(all_experiments, modality, output_dir, name, 'nrmse')

    if has_mse_keys(all_experiments, modality):
        _save_metric_summary(all_experiments, modality, output_dir, name, 'mse')
    else:
        print('\n[info] MSE keys not found in results — skipping MSE outputs.')
        print('       Re-run experiments to generate metrics.pt with MSE data.')

    metrics = ['nrmse'] + (['mse'] if has_mse_keys(all_experiments, modality) else [])

    if has_noise_variation(all_experiments):
        for metric in metrics:
            _save_grouped_metric_summary(all_experiments, modality, output_dir, name, metric, 'noise')
    else:
        print('\n[info] Single SNR level detected — skipping by-SNR table.')

    if has_sampling_variation(all_experiments):
        for metric in metrics:
            _save_grouped_metric_summary(all_experiments, modality, output_dir, name, metric, 'sampling')
    else:
        print('\n[info] Single measurement count detected — skipping by-sampling table.')
