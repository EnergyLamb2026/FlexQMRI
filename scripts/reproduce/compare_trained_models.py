'''Compare results from multiple fitting/training experiments.

This script reads pre-computed metrics.pt files saved by run_lsq,
train_networks, or train_multi_fixed_models and produces:
- A raw .pt file with all per-sample errors keyed by label
- A summary table printed to the terminal and saved as CSV
- Per-parameter boxplots saved as PNG

The recommended workflow is to write a .sh file that:
1. Assigns a descriptive --global_run_id to each training command
2. Passes all those IDs to this script at the end via --global_run_ids

Example .sh file (see compare_experiments.sh for a full template):

    COMP="lsq_vs_mlp_$(date +%Y%m%d)"
    python -m scripts.run_lsq -c config/lsq.yaml --global_run_id ${COMP}_lsq
    python -m scripts.train_networks -c config/mlp.yaml --global_run_id ${COMP}_mlp
    python -m scripts.train_networks -c config/mlp.yaml --global_run_id ${COMP}_mlp_bn --batchnorm
    python -m scripts.compare_experiments \
        -c config/mlp.yaml --data_config_path config/ivim_data.yaml \
        --global_run_ids ${COMP}_lsq ${COMP}_mlp ${COMP}_mlp_bn --name $COMP

Results are discovered automatically by scanning results/ for directories matching
each global_run_id and merging all metrics.pt files found inside (for multi-seed runs).
'''
from pathlib import Path

from flexqmri.evaluation.comparisons import load_experiments, run_comparison
from flexqmri.utils.config import load_and_validate_config
from flexqmri.utils.parse import parse_compare_run_args


def main():
    """Discover results by global_run_id, then produce summary table, CSV, and boxplots."""
    args = parse_compare_run_args()
    config, _, _ = load_and_validate_config(args)

    modality = config['data']['modality']
    results_root = config['paths']['output_dir']
    output_dir = Path(results_root) / modality / 'comparisons' / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_experiments = load_experiments(args.global_run_ids, results_root, modality)
    run_comparison(all_experiments, modality, output_dir, args.name)


if __name__ == '__main__':
    main()
