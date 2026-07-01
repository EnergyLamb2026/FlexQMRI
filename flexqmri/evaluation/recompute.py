"""Recompute missing ``metrics.pt`` for trained-model folders.

When :mod:`scripts.reproduce.compare_trained_models` discovers a run folder that
holds a saved model (or an LSQ config) but no ``metrics.pt``, these helpers
regenerate the test set from the folder's ``config.json``, evaluate, and write
``metrics.pt`` so the comparison can proceed.

The test set is regenerated from the config's seed via the same
``get_dataset_loaders`` pipeline training used, so it follows the same
distribution and split as the original run.

Supported per ``config['train']['model']``:
- ``mlp`` / ``transformer``: multi-fixed run; one model per length over the full
  ``simulation.min_x_length..max_x_length`` range. The variable-length dataset is
  simulated once and then filtered to each length (not regenerated per length); each
  model is tested on the config's test split of its length's samples. Raises if any
  length is missing.
- ``ncde``: single variable-length model. Raises if ``train.adaptive`` is True,
  because the decayed ODE tolerances used at the original test time are not persisted.
- anything else (``trf`` / ``lm``): LSQ; re-runs the scipy fit on the test set.
"""

import copy
import re
from pathlib import Path

import torch

from flexqmri.dataset import DataLoaderFactory, create_dataset_instance, get_dataset_loaders
from flexqmri.evaluation import evaluate
from flexqmri.evaluation import utils as eval_utils
from flexqmri.models import lsq_fit
from flexqmri.networks import utils as net_utils
from flexqmri.utils.config import load_config_for_model
from flexqmri.utils.utils import set_seed

_SEED_RE = re.compile(r'_seed(\d+)')
_LENGTH_RE = re.compile(r'best_model_(\d+)\.pth')


def _seed_for_folder(config: dict, folder: Path) -> int:
    """Return the run seed, mirroring how training seeds each seed run.

    Args:
        config (dict): Folder config (provides ``train.seed`` base seed).
        folder (Path): Run folder; a ``_seed{N}`` suffix selects the seed offset.

    Returns:
        int: ``train.seed + N`` (N defaults to 0 when no suffix is present).
    """
    match = _SEED_RE.search(folder.name)
    return config['train']['seed'] + (int(match.group(1)) if match else 0)


def _seeded_test_loader(config: dict, generator: torch.Generator) -> torch.utils.data.DataLoader:
    """Build the test DataLoader for a config, reusing the given generator.

    Args:
        config (dict): Configuration dictionary for data generation.
        generator (torch.Generator): Generator driving data generation and split.

    Returns:
        torch.utils.data.DataLoader: The test loader from get_dataset_loaders.
    """
    _, _, test_loader = get_dataset_loaders(config=config, generator=generator)
    return test_loader


def _expected_lengths(config: dict) -> list:
    """Return the full length range a multi-fixed run is expected to cover.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        list[int]: ``range(min_x_length, max_x_length + 1)`` from the simulation config.
    """
    sim = config['data']['simulation']
    return list(range(sim['min_x_length'], sim['max_x_length'] + 1))


def _present_lengths(folder: Path) -> list:
    """Return the sorted lengths for which a ``best_model_{length}.pth`` exists.

    Args:
        folder (Path): Run folder to scan.

    Returns:
        list[int]: Sorted lengths discovered in the folder.
    """
    return sorted(int(_LENGTH_RE.search(p.name).group(1)) for p in folder.glob('best_model_*.pth'))


def _recompute_lsq(folder: Path, config: dict, modality: str) -> None:
    """Re-run the LSQ fit on a freshly generated test set and save metrics.pt.

    Args:
        folder (Path): Run folder containing config.json.
        config (dict): Folder configuration.
        modality (str): Data modality.
    """
    seed = _seed_for_folder(config, folder)
    set_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    test_loader = _seeded_test_loader(config, generator)
    results = lsq_fit.fit_loader_from_config(test_loader, config)
    evaluate.finalize_and_save_metrics(results, folder, modality, seed=seed)


def _recompute_ncde(folder: Path, config: dict, modality: str) -> None:
    """Test the saved NCDE on a freshly generated test set and save metrics.pt.

    Args:
        folder (Path): Run folder containing best_model_0.pth and config.json.
        config (dict): Folder configuration.
        modality (str): Data modality.

    Raises:
        ValueError: If ``train.adaptive`` is True (decayed ODE tolerances are not persisted).
    """
    if config['train'].get('adaptive', False):
        raise ValueError(
            f"Cannot recompute metrics for adaptive NCDE in {folder}: the decayed ODE "
            "tolerances used at original test time are not persisted. Re-run training."
        )
    seed = _seed_for_folder(config, folder)
    set_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    test_loader = _seeded_test_loader(config, generator)
    model = net_utils.load_model(folder / 'best_model_0.pth', config)
    results = eval_utils.test_network(
        config, test_loader, model, modality, device='cpu', atol=None, rtol=None
    )
    evaluate.finalize_and_save_metrics(results, folder, modality, seed=seed)


def _recompute_multi_fixed(folder: Path, config: dict, modality: str) -> None:
    """Test every fixed-length model in a multi-fixed folder and save aggregated metrics.pt.

    Simulates the whole variable-length dataset once (``fixed_length = 0``), then for each
    length filters it to that length and takes the test split (the config's
    ``train_val_test_split``) via ``DataLoaderFactory`` — the same code path as training,
    minus the regeneration — testing the matching model on it. A sample of length L drawn in
    variable mode is distributionally identical to one drawn in fixed-L mode, so each model
    is evaluated on data consistent with its training. Writes a ``metrics_{length}.pt`` per
    length and an aggregated ``metrics.pt``.

    Args:
        folder (Path): Run folder containing best_model_{length}.pth files and config.json.
        config (dict): Folder configuration.
        modality (str): Data modality.

    Raises:
        ValueError: If the folder does not contain a model for every expected length.
    """
    expected = _expected_lengths(config)
    present = _present_lengths(folder)
    if present != expected:
        raise ValueError(
            f"Incomplete multi-fixed run in {folder}: expected models for lengths {expected}, "
            f"found {present}. Re-run train_multi_fixed_models for the missing lengths."
        )

    seed = _seed_for_folder(config, folder)
    set_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    gen_config = copy.deepcopy(config)
    gen_config['data']['fixed_length'] = 0
    X, y, noise = create_dataset_instance(gen_config, modality, 'simulation').generate_data(generator)

    full_results = eval_utils.init_results_dict(modality)
    for length in expected:
        config_length = copy.deepcopy(config)
        config_length['data']['fixed_length'] = length

        _, _, test_loader = DataLoaderFactory(config_length).create_loaders(X, y, noise, generator)
        model = net_utils.load_model(folder / f'best_model_{length}.pth', config_length)
        results_length = eval_utils.test_network(
            config_length, test_loader, model, modality, device='cpu', atol=None, rtol=None
        )

        evaluate.finalize_and_save_metrics(
            results_length, folder, modality,
            metrics_filename=f'metrics_{length}.pt', print_results=False,
        )
        eval_utils.accumulate_results(full_results, results_length)

    evaluate.finalize_and_save_metrics(
        full_results, folder, modality, seed=seed,
        scatter_filename='scatter_pred_vs_true_aggregated.png',
    )


def recompute_folder(folder: Path) -> None:
    """Regenerate ``metrics.pt`` for a single run folder, dispatching by model type.

    Args:
        folder (Path): Run folder containing a ``config.json`` and a saved model
            (or, for LSQ, just the config).

    Raises:
        ValueError: Propagated from the per-type recompute (e.g. adaptive NCDE,
            incomplete multi-fixed run).
    """
    config = load_config_for_model(folder / 'config.json')
    model_name = config['train']['model']
    modality = config['data']['modality']

    if model_name in ('mlp', 'transformer'):
        _recompute_multi_fixed(folder, config, modality)
    elif model_name == 'ncde':
        _recompute_ncde(folder, config, modality)
    else:
        _recompute_lsq(folder, config, modality)


def ensure_metrics(global_run_id: str, results_root: str = 'results', modality: str = None) -> None:
    """Recompute ``metrics.pt`` for any folder under ``global_run_id`` that lacks it.

    Mirrors the discovery scope of ``get_computed_metrics`` so every folder the
    comparison would read is guaranteed to have a metrics file afterward.

    Args:
        global_run_id (str): The run ID assigned to one experiment.
        results_root (str): Root directory to search (default: 'results').
        modality (str, optional): If given, restrict the search to the
            ``{modality}/`` subtree (so a run ID shared across modalities is not
            matched in the wrong one). Defaults to None (search all modalities).
    """
    root = Path(results_root)
    prefix = f'{modality}/' if modality else ''
    for config_path in sorted(root.glob(f'{prefix}**/{global_run_id}/**/config.json')):
        folder = config_path.parent
        if (folder / 'metrics.pt').exists():
            continue
        print(f"  [recompute] no metrics.pt in {folder} — regenerating from saved model/config ...")
        recompute_folder(folder)
