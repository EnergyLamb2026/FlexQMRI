'''Functions to compute IVIM parameter maps from patient data.'''

import os
import time

from flexqmri.utils.config import load_config_for_model

from flexqmri.imaging.structures import PatientData
from flexqmri.imaging.patient import patient_data_to_loader, predictions_to_maps
from flexqmri.imaging.savers import save_maps

from flexqmri.compute_maps.general import process_fitting

def compute_ivim_maps(
    config: dict,
    patient_data: PatientData,
    patient_id: str,
    study: str,
    pipeline: str,
) -> dict:
    """Compute IVIM parameter maps from patient data and save them if configured.

    Assumes the data in ``patient_data`` is already preprocessed (normalized,
    denoised, aligned to anatomical reference).

    Args:
        config (dict): Configuration dictionary. Must contain
            ``paths.ivim_deepmr_config_path``, ``paths.ivim_model_path``,
            ``paths.output_data``, and ``ivim.maps.save``.
        patient_data (PatientData): Patient IVIM data with metadata.
        patient_id (str): Patient identifier, used to build the output path.
        study (str): Study/session identifier, used to build the output path.
        pipeline (str): Pipeline label (e.g. 'standard'), used to build the output path.

    Returns:
        tuple[dict, str, float]: ``{'f', 'D', 'D_star'}`` maps each of shape
            (x, y, z), the model name, and the CPU time per voxel in seconds.
            For LSQ methods the wall-clock time is multiplied by
            ``os.cpu_count()`` to approximate total CPU time.

    Notes:
        IVIM predictions are ordered (S0, f, D, D*) at indices 0–3.
    """
    # 1 - Extract data
    deepmr_config = load_config_for_model(config['paths']['ivim_deepmr_config_path'])
    deepmr_config['data']['fixed_length'] = patient_data.get_length()
    deepmr_config['data']['param_ranges'] = config['ivim'].get('param_ranges')
    model_name = deepmr_config['train']['model']
    model_path = config['paths']['ivim_model_path']
    data_loader = patient_data_to_loader(patient_data, deepmr_config) # Create a DataLoader from PatientData
    ivim_maps = {}

    # 2 - Fit IVIM model using specified method
    start = time.time()
    results = process_fitting(data_loader, deepmr_config, 'ivim', model_name, model_path)

    # 3 - Extract the data
    predictions_4d = predictions_to_maps(results, patient_data)
    n_voxels = len(results['predictions'])
    n_cores = os.cpu_count() if model_name in ('lm', 'trf') else 1
    time_per_voxel = (time.time() - start) * n_cores / n_voxels
    print(f"IVIM fitting complete. Parameter maps shape: {predictions_4d.shape}. CPU time/voxel: {time_per_voxel:.6f} seconds.")

    ivim_maps['f'] = predictions_4d[:, :, :, 1]
    ivim_maps['D'] = predictions_4d[:, :, :, 2]
    ivim_maps['D_star'] = predictions_4d[:, :, :, 3]

    if config['ivim']['maps']['save']:
        save_maps(ivim_maps, patient_data, model_name, config['paths']['output_data'], patient_id, study, pipeline, 'ivim')

    return ivim_maps, model_name, time_per_voxel
