'''Functions to compute R2* parameter maps from patient data.'''

import os
import time

import numpy as np

from flexqmri.utils.config import load_config_for_model

from flexqmri.imaging.structures import PatientData
from flexqmri.imaging.patient import patient_data_to_loader, predictions_to_maps
from flexqmri.imaging.savers import save_maps

from flexqmri.compute_maps.general import process_fitting

def compute_r2star_map(
    config: dict,
    patient_data: PatientData,
    patient_id: str,
    study: str,
    pipeline: str,
) -> dict:
    """Compute R2* maps from multi-echo T2* patient data.

    Fits a T2* decay model via process_fitting, then inverts T2* to obtain R2* = 1/T2*.
    Saves maps if configured.

    Args:
        config (dict): Configuration dictionary. Must contain
            ``paths.r2star_deepmr_config_path``, ``paths.r2star_model_path``,
            ``paths.output_data``, and ``t2star.maps.save``.
        patient_data (PatientData): Patient multi-echo data with metadata.
        patient_id (str): Patient identifier, used to build the output path.
        study (str): Study/session identifier, used to build the output path.
        pipeline (str): Pipeline label (e.g. 'standard'), used to build the output path.

    Returns:
        tuple[dict, str, float]: ``{'R2star': np.ndarray}`` of shape (x, y, z),
            the model name, and the CPU time per voxel in seconds.
            For LSQ methods the wall-clock time is multiplied by
            ``os.cpu_count()`` to approximate total CPU time.

    Notes:
        T2* predictions are at index 1 (after S0 at index 0).
        T2* model output is in ms; R2* = 1000/T2* is in s⁻¹ to match R2 units.
        Voxels where T2* == 0 are set to R2* = 0 to avoid division by zero.
    """
    # 1 - Extract data
    deepmr_config = load_config_for_model(config['paths']['r2star_deepmr_config_path'])
    deepmr_config['data']['fixed_length'] = patient_data.get_length()
    deepmr_config['data']['param_ranges'] = config['t2star'].get('param_ranges')
    model_name = deepmr_config['train']['model']
    model_path = config['paths']['r2star_model_path']
    data_loader = patient_data_to_loader(patient_data, deepmr_config)

    # 2 - Fit to extract T2* maps
    start = time.time()
    results = process_fitting(data_loader, deepmr_config, 't2star', model_name, model_path)
    predictions_4d = predictions_to_maps(results, patient_data)
    n_voxels = len(results['predictions'])
    n_cores = os.cpu_count() if model_name in ('lm', 'trf') else 1
    time_per_voxel = (time.time() - start) * n_cores / n_voxels
    print(f"T2* fitting complete. Shape: {predictions_4d.shape}. CPU time/voxel: {time_per_voxel:.6f} seconds.")

    # 3 - Invert T2* (ms) to obtain R2* (s⁻¹); zero out non-tissue voxels
    t2star_map = predictions_4d[:, :, :, 1]
    r2star_map = np.where(t2star_map > 0, 1000.0 / t2star_map, 0.0)
    r2star_maps = {'R2star': r2star_map}

    # 4 - Save maps
    if config['t2star']['maps']['save']:
        save_maps(r2star_maps, patient_data, model_name, config['paths']['output_data'], patient_id, study, pipeline, 'r2star')

    return r2star_maps, model_name, time_per_voxel

def compute_r2star_map_from_t2star(
    config: dict,
    patient_data: PatientData,
    patient_id: str,
    study: str,
    pipeline: str,
) -> dict:
    """Load a precomputed R2* map from patient data and save it if configured.

    The vendor T2* map is already inverted; this function loads it as-is and
    applies the mask when present to zero out non-tissue voxels.

    Args:
        config (dict): Configuration dictionary. Must contain ``paths.output_data``
            and ``t2star.maps.save``.
        patient_data (PatientData): Patient T2* data. ``data`` must be a 3D
            array of shape (x, y, z) holding the precomputed R2* map.
            ``mask_array``, if provided, is a binary array of the same shape.
        patient_id (str): Patient identifier, used to build the output path.
        study (str): Study/session identifier, used to build the output path.
        pipeline (str): Pipeline label (e.g. 'standard'), used to build the output path.

    Returns:
        dict: ``{'R2star': np.ndarray}`` of shape (x, y, z).

    Notes:
        The vendor map is already inverted (R2* in s⁻¹); it is loaded as-is.
    """
    r2star = patient_data.data.astype(np.float32)

    if patient_data.mask_array is not None:
        r2star = np.where(patient_data.mask_array > 0, r2star, 0.0)
    r2star_maps = {'R2star': r2star}

    if config['t2star']['maps']['save']:
        save_maps(r2star_maps, patient_data, 'vendor', config['paths']['output_data'], patient_id, study, pipeline, 'r2star')

    return r2star_maps
