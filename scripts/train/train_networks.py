'''Script to train a supervised regression model on synthetic MRI data.'''

import time
import uuid

import torch

from flexqmri.models.supervised import run_sup_network_seeds
from flexqmri.evaluation import utils as eval_utils
from flexqmri.utils import parse
from flexqmri.utils import config

if __name__ == '__main__':

    args = parse.parse_args()
    config, seed_nbr, base_seed = config.load_and_validate_config(args)
    global_run_id = args.global_run_id or str(uuid.uuid4())[:8]

    print(f"Training a supervised network with architecture from : {args.model_config_path}")
    print(f"Global run ID: {global_run_id}")
    start = time.time()

    all_results = run_sup_network_seeds(
        config, seed_nbr, base_seed, global_run_id
    )

    elapsed = time.time() - start

    for metrics_path in eval_utils.get_computed_metrics(global_run_id, 'results'):
        data = torch.load(metrics_path, weights_only=False)
        data['training_time_seconds'] = elapsed
        torch.save(data, metrics_path)

    if seed_nbr > 1:
        aggregated_results = eval_utils.aggregate_results_across_seeds(all_results, seed_nbr, seed=base_seed)
    else:
        aggregated_results = all_results[0]

    print(f'\nTotal training time: {elapsed:.2f} seconds.')