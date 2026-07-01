'''Shared fitting dispatch used to compute qMRI parameter maps.'''

from flexqmri.evaluation import utils as eval_utils
from flexqmri.networks.utils import load_model
from flexqmri.models.lsq_fit import fit_loader_from_config

def process_fitting(
    data_loader,
    deepmr_config: dict,
    modality: str,
    model_name: str,
    model_path: str,
) -> dict:
    """Dispatch fitting to the appropriate method (LSQ or neural network).

    Routes inference to ``fit_loader_from_config`` for LSQ methods (``lm``,
    ``trf``) or to ``eval_utils.test_network`` for neural network methods
    (``mlp``, ``transformer``, ``ncde``).

    Args:
        data_loader: PyTorch DataLoader produced by ``patient_data_to_loader``.
        deepmr_config (dict): FlexQMRI config loaded from a model JSON file.
        modality (str): Modality label forwarded to ``test_network`` (e.g. ``'ivim'``, ``'t2star'``).
        model_name (str): Model identifier read from ``deepmr_config['train']['model']``.
        model_path (str): Path to the saved model weights (.pth). Only used for neural network methods.

    Returns:
        dict: Raw results dict containing at least ``'predictions'``: a list of
            per-voxel tensors of shape (n_params,).

    Raises:
        ValueError: If ``model_name`` is not one of the supported LSQ or neural network identifiers.
    """
    if model_name in ('lm', 'trf'):
        print(f"Fitting {modality} model using LSQ method '{model_name}'.")
        results = fit_loader_from_config(data_loader, deepmr_config)
    elif model_name in ('mlp', 'transformer', 'ncde'):
        print(f"Fitting {modality} model using neural network method '{model_name}'.")
        model = load_model(model_path, deepmr_config)
        results = eval_utils.test_network(
            config=deepmr_config,
            test_loader=data_loader,
            model=model,
            modality=modality,
            device='cpu',
        )
    else:
        raise ValueError(f"Unsupported model type '{model_name}' specified in config.")

    return results
