'''Run LSQ fitting on synthetic data using scipy curve_fit.

This script performs voxel-wise least-squares fitting (TRF or LM method)
parallelized via joblib. It supports multi-seed runs with bootstrap CIs.

Usage:
    python -m scripts.run_lsq -c config/lsq.yaml [-d /path/to/data]

The fitted results are saved to:
    results/{modality}/lsq/{global_run_id}/{run_id}/metrics.pt
'''
import datetime
import json
import time
import uuid
from pathlib import Path

import torch

from flexqmri.dataset import get_dataset_loaders
import flexqmri.models.lsq_fit as lsq
import flexqmri.evaluation.utils as eval_utils
import flexqmri.evaluation.results_plots as results_plots
from flexqmri.utils import parse
from flexqmri.utils import config as config_utils
from flexqmri.utils.io import get_model_path
from flexqmri.utils.utils import make_serializable, set_seed


def fit_lsq_network(config: dict, seed_run: int = 0):
    """
    Fit LSQ model on synthetic IVIM data for a single seed run.

    Args:
        config (dict): Configuration dictionary containing model and training parameters
        seed_run (int): Current seed run number (for multiple runs)

    Returns:
        dict: Results dictionary with errors for each parameter
    """
    
    # Create generator for reproducibility with seed offset for multiple runs
    generator = torch.Generator()
    base_seed = config["train"]["seed"]
    current_seed = base_seed + seed_run
    generator.manual_seed(current_seed)

    # Load data using new OOP API
    train_loader, val_loader, test_loader = get_dataset_loaders(
        config=config,
        generator=generator
    )

    # Fit model
    print(f'Starting LSQ {config["train"]["model"]} fitting on test data ({len(test_loader.dataset)} samples)...')

    results = lsq.fit_loader_from_config(test_loader, config)
    spearman_r = eval_utils.compute_spearman_r(results, config['data']['modality'])
    results['spearman_r'] = spearman_r
    eval_utils.print_fit_results(results, config['data']['modality'])

    return results


def run_lsq_multiple_seeds(config: dict, seed_nbr: int, base_seed: int) -> list:
    """Run LSQ fitting for multiple seeds and aggregate results."""
    all_results = []
    
    for seed_run in range(seed_nbr):
        print(f"\n{'='*80}")
        print(f"LSQ fitting run {seed_run + 1}/{seed_nbr}")
        print(f"{'='*80}")
        
        # Set seeds for reproducibility
        current_seed = base_seed + seed_run
        set_seed(current_seed)
        
        results = fit_lsq_network(config, seed_run=seed_run)
        all_results.append(results)
    
    return all_results


def save_lsq_results(results: dict, config: dict, run_id: str, global_run_id: str) -> Path:
    """Save LSQ results to a metrics.pt file.

    Args:
        results (dict): Results dictionary from fitting.
        config (dict): Configuration dictionary.
        run_id (str): Unique identifier for this run.
        global_run_id (str): Parent run ID for grouping runs.

    Returns:
        Path: Path to the saved metrics file.
    """
    modality = config['data']['modality']
    model_type = config['train']['model']  # e.g. 'lsq_trf', 'lsq_lm'
    metrics_path = get_model_path(model_type, modality, run_id, filename='metrics.pt', global_run_id=global_run_id)
    torch.save(results, metrics_path)

    config_path = metrics_path.parent / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(make_serializable(config), f, indent=2)

    return metrics_path


if __name__ == '__main__':
    args = parse.parse_args()

    config, seed_nbr, base_seed = config_utils.load_and_validate_config(args)
    
    global_run_id = args.global_run_id or str(uuid.uuid4())[:8]

    print(f"Global run ID: {global_run_id}")
    
    start = time.time()
    
    # Run LSQ fitting loop
    all_results = run_lsq_multiple_seeds(config, seed_nbr, base_seed)
    
    elapsed = time.time() - start

    # Save results for each seed
    for seed_run, results in enumerate(all_results):
        results['training_time_seconds'] = elapsed
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{seed_run}"
        metrics_path = save_lsq_results(results, config, run_id, global_run_id)
        print(f'Metrics saved to {metrics_path}')

        scatter_path = metrics_path.parent / 'scatter_pred_vs_true.png'
        results_plots.plot_scatter_predictions(
            results, config['data']['modality'],
            spearman_r=results.get('spearman_r'),
            save_path=str(scatter_path),
        )
    
    # Aggregate results across seeds if multiple runs
    if seed_nbr > 1:
        aggregated_results = eval_utils.aggregate_results_across_seeds(all_results, seed_nbr, seed=base_seed)
    else:
        aggregated_results = all_results[0]

    print(f'\nTotal LSQ fitting time: {elapsed:.2f} seconds.')