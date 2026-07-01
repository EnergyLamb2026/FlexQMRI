"""Shared post-evaluation helper: attach metrics, persist, and plot.

Factors out the ``compute Spearman -> (print) -> save metrics .pt -> save
scatter`` block shared by training (:mod:`flexqmri.models.supervised`) and
metrics recomputation (:mod:`flexqmri.evaluation.recompute`).
"""

from pathlib import Path

import torch

from flexqmri.evaluation import results_plots
from flexqmri.evaluation import utils as eval_utils


def finalize_and_save_metrics(
    results: dict,
    model_dir: Path,
    modality: str,
    *,
    seed: int = 42,
    metrics_filename: str = 'metrics.pt',
    scatter_filename: str = 'scatter_pred_vs_true.png',
    print_results: bool = True,
) -> dict:
    """Attach Spearman ρ to results, save the metrics file, and write the scatter plot.

    Args:
        results (dict): Per-sample results from ``test_network`` or LSQ fitting,
            containing 'predictions' and 'y_true' (required by Spearman).
        model_dir (Path): Directory to write the metrics file and scatter plot into.
        modality (str): Data modality ('ivim', 't2star').
        seed (int, optional): Seed for the bootstrap CIs in Spearman/printing. Defaults to 42.
        metrics_filename (str, optional): Filename for the saved metrics. Defaults to 'metrics.pt'.
        scatter_filename (str, optional): Filename for the scatter plot. Defaults to 'scatter_pred_vs_true.png'.
        print_results (bool, optional): Whether to print fit results with CIs. Defaults to True.

    Returns:
        dict: ``results`` with 'spearman_r' attached.

    Notes:
        The metrics file is what ``compare_trained_models`` discovers; the scatter
        plot is a side artifact mirroring the training scripts' behavior.
    """
    model_dir = Path(model_dir)
    spearman_r = eval_utils.compute_spearman_r(results, modality, seed=seed)
    results['spearman_r'] = spearman_r

    if print_results:
        eval_utils.print_fit_results(results, modality, seed=seed)

    metrics_path = model_dir / metrics_filename
    torch.save(results, metrics_path)
    print(f'Metrics saved to {metrics_path}')

    results_plots.plot_scatter_predictions(
        results, modality, spearman_r=spearman_r,
        save_path=str(model_dir / scatter_filename),
    )
    return results
