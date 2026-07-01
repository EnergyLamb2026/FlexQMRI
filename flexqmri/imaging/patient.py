'''Functions to load MRI data and convert patient data to DataLoaders.'''

import copy
from typing import Optional

import numpy as np
import torch
import flexqmri.imaging.loaders
from flexqmri.imaging.structures import PatientData

from flexqmri.dataset import get_dataset_loaders

def load_ivim_patient_data(
    study_path: str,
    all_series: list,
    config: dict,
    patient_id: str,
    slice_idx: int = None,
) -> PatientData:
    """Load IVIM data from NIfTI files into a PatientData object.

    Filters the provided series list to identify IVIM series and loads the
    corresponding NIfTI offsets from disk. Filter keywords are read from
    ``config['ivim']['loading']``.

    Args:
        study_path (str): Path to the study directory.
        all_series (list): List of all series present in the study.
        config (dict): Configuration dictionary containing loading filters and offset_path keys under ``config['ivim']``.
        patient_id (str): Patient identifier.
        slice_idx (int, optional): Slice index to extract. Keeps all slices
            when ``None``. Defaults to ``None``.

    Returns:
        PatientData: Object containing IVIM data, mask, affine and metadata.

    Raises:
        ValueError: If no IVIM series match the filters in config.
    """
    loading = config['ivim']['loading']
    return flexqmri.imaging.loaders._load_offset_patient_data(
        study_path=study_path,
        all_series=all_series,
        patient_id=patient_id,
        keywords=loading['keywords'],
        include=loading['include'],
        exclude=loading['exclude'],
        offset_path=config['ivim']['loading']['offset_path'],
        serie_name='IVIM',
        slice_idx=slice_idx,
    )


def load_t2star_patient_data(
    study_path: str,
    all_series: list,
    config: dict,
    patient_id: str,
    slice_idx: int = None,
) -> PatientData:
    """Load T2* data from NIfTI files into a PatientData object.

    Filters the provided series list to identify T2* (Images) series and loads
    the corresponding NIfTI echo-time volumes from disk. Filter keywords are
    read from ``config['t2star']['loading']``.

    Args:
        study_path (str): Path to the study directory.
        all_series (list): List of all series present in the study.
        config (dict): Configuration dictionary containing loading filters and offset_path keys under ``config['t2star']``.
        patient_id (str): Patient identifier.
        slice_idx (int, optional): Slice index to extract. Keeps all slices
            when ``None``. Defaults to ``None``.

    Returns:
        PatientData: Object containing T2* data, mask, affine and metadata.
            ``x`` holds the echo times in ms as read from the offset file.

    Raises:
        ValueError: If no T2* series match the filters in config.
    """
    loading = config['t2star']['loading']
    return flexqmri.imaging.loaders._load_offset_patient_data(
        study_path=study_path,
        all_series=all_series,
        patient_id=patient_id,
        keywords=loading['keywords'],
        include=loading['include'],
        exclude=loading['exclude'],
        offset_path=loading['offset_path'],
        serie_name='T2star',
        slice_idx=slice_idx,
    )

def patient_data_to_loader(patient_data: PatientData, deepmr_config: dict,
                            generator: Optional[torch.Generator] = None):
    """Convert PatientData to a test DataLoader for FlexQMRI inference.

    Flattens the 4D spatial volume (x, y, z, n_meas) to (n_voxels, n_meas)
    using the mask, then delegates to FlexQMRI's get_dataset_loaders
    which formats the signals into the (n_samples, n_meas, 2) path format
    expected by all model types.

    Args:
        patient_data (PatientData): Patient MRI data. data must be 4D
            (x, y, z, n_measurements); mask_array selects valid voxels.
            If mask_array is None, all voxels are used.
        deepmr_config (dict): FlexQMRI config built by _build_deepm_config.
            Must contain 'data.invivo.b_values' and 'data.invivo.n_samples'.
        generator (torch.Generator, optional): PyTorch generator for reproducibility.

    Returns:
        torch.utils.data.DataLoader: Test loader ready for FlexQMRI inference.
            The train and val loaders are discarded (split is [0, 0, 1] for invivo).
    """
    data_4d = patient_data.data  # (x, y, z, n_meas)
    mask = patient_data.mask_array  # (x, y, z) or None

    # Update deepmr_config with b-values from patient_data.x (if not already set) and ensure invivo section exists
    deepmr_config = copy.deepcopy(deepmr_config)
    if deepmr_config['data'].get('invivo') is None:
        deepmr_config['data']['invivo'] = {}
    deepmr_config['data']['invivo']['x'] = list(patient_data.x)

    if mask is not None:
        X_flat = data_4d[mask > 0].astype(np.float32)  # (n_voxels, n_meas)
    else:
        X_flat = data_4d.reshape(-1, data_4d.shape[-1]).astype(np.float32)

    deepmr_config['data']['invivo']['n_samples'] = len(X_flat)

    # Ensure all voxels go to the test loader; the simulation split (from training YAMLs) must not apply to in-vivo inference.
    if deepmr_config['data'].get('simulation') is None:
        deepmr_config['data']['simulation'] = {}
    deepmr_config['data']['simulation']['train_val_test_split'] = [0.0, 0.0, 1.0]

    _, _, test_loader = get_dataset_loaders(deepmr_config, generator=generator, X=X_flat)
    return test_loader


def predictions_to_maps(results: dict, patient_data: PatientData) -> np.ndarray:
    """Reshape flat voxel predictions back into a 4D spatial parameter volume.

    Args:
        results (dict): Output of test_network or fit_loader. Must contain
            'predictions' as a list of tensors of shape (n_params,), one per
            valid voxel in the same order as patient_data_to_loader produced them.
        patient_data (PatientData): Source patient data. data must be 4D
            (x, y, z, n_measurements); mask_array selects valid voxels.

    Returns:
        np.ndarray: Parameter maps of shape (x, y, z, n_params), zero-filled
            outside the mask.
    """
    predictions = torch.stack(results['predictions']).cpu().numpy()  # (n_voxels, n_params)
    spatial_shape = patient_data.data.shape[:3]
    n_params = predictions.shape[1]
    mask = patient_data.mask_array

    if mask is not None:
        predictions_4d = np.zeros((*spatial_shape, n_params), dtype=np.float32)
        predictions_4d[mask > 0] = predictions
    else:
        predictions_4d = predictions.reshape(*spatial_shape, n_params).astype(np.float32)

    return predictions_4d
