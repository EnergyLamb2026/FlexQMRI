import os

import numpy as np
from dipy.io.image import save_nifti

from flexqmri.imaging.structures import PatientData


def save_maps(
    maps: dict,
    patient_data: PatientData,
    model_name: str,
    output_dir: str,
    patient_id: str,
    study: str,
    pipeline: str,
    maps_name: str,
) -> None:
    """Save parameter maps to NIfTI files under <output_dir>/<patient_id>/<study>/<pipeline>/<maps_name>/<model_name>/.

    Args:
        maps (dict): Map name to shape (x, y, z) numpy array.
        patient_data (PatientData): Must contain affine, original_shape, and slice_idx.
        model_name (str): Model name used for fitting, last level of the output path.
        output_dir (str): Root output directory (config['paths']['output_data']).
        patient_id (str): Patient identifier.
        study (str): Study/session identifier.
        pipeline (str): Pipeline label (e.g. 'standard').
        maps_name (str): Map type identifier (e.g. 'ivim', 'r2star').

    Returns:
        None
    """
    maps_path = os.path.join(output_dir, patient_id, study, pipeline, maps_name, model_name)
    print(f"Saving maps to: {maps_path}")
    os.makedirs(maps_path, exist_ok=True)
    for key, map_data in maps.items():
        if patient_data.slice_idx is not None:
            full = np.full(patient_data.original_shape[:3], np.nan)
            full[:, :, patient_data.slice_idx] = map_data[:, :, 0]
            map_data = full
        save_nifti(os.path.join(maps_path, f"{key}.nii.gz"), map_data, patient_data.affine)


def save_mask(
    mask: np.ndarray,
    patient_data: PatientData,
    output_dir: str,
    patient_id: str,
    study: str,
    pipeline: str,
    maps_name: str,
) -> None:
    """Save the in-ROI mask alongside parameter maps.

    Args:
        mask (np.ndarray): Boolean or integer mask of shape (x, y, z).
        patient_data (PatientData): Must contain affine, original_shape, and slice_idx.
        output_dir (str): Root output directory (config['paths']['output_data']).
        patient_id (str): Patient identifier.
        study (str): Study/session identifier.
        pipeline (str): Pipeline label (e.g. 'standard').
        maps_name (str): Map type identifier; mask is saved one level above model folders.

    Returns:
        None

    Notes:
        Written to ``<output_dir>/<patient_id>/<study>/<pipeline>/<maps_name>/mask.nii.gz``.
        The mask is shared by every model under that maps_name, so it is saved once
        per map type rather than per model.
    """
    save_dir = os.path.join(output_dir, patient_id, study, pipeline, maps_name)
    os.makedirs(save_dir, exist_ok=True)
    mask_out = mask.astype(np.uint8)
    if patient_data.slice_idx is not None:
        full = np.zeros(patient_data.original_shape[:3], dtype=np.uint8)
        full[:, :, patient_data.slice_idx] = mask_out[:, :, 0]
        mask_out = full
    out_path = os.path.join(save_dir, "mask.nii.gz")
    save_nifti(out_path, mask_out, patient_data.affine)
    print(f"Saving mask to: {out_path}")


def save_nifti_bvalues(data_array: np.ndarray, patient_data: PatientData) -> None:
    """Save each b-value volume of an IVIM series in its corresponding path.

    Args:
        data_array (np.ndarray): Shape (x, y, z, n_bvalues), data to save.
        patient_data (PatientData): Must contain study, slice_idx, affine, and x (b-values).

    Returns:
        None
    """
    save_folder = os.path.join(patient_data.study, "ivim_data")
    os.makedirs(save_folder, exist_ok=True)
    print(f"Saving IVIM b-value volumes to: {save_folder}")
    for idx, bvalue in enumerate(patient_data.x):
        bvalue_data = data_array[:, :, :, idx]
        nifti_filename = os.path.join(save_folder, f"{bvalue}_s_mm2.nii.gz")
        save_nifti(nifti_filename, bvalue_data, patient_data.affine)


def save_nifti_tes(data_array: np.ndarray, patient_data: PatientData) -> None:
    """Save each echo-time volume of an R2* series in its corresponding path.

    Args:
        data_array (np.ndarray): Shape (x, y, z, n_echoes), data to save.
        patient_data (PatientData): Must contain study, slice_idx, affine, and x (echo times in ms).

    Returns:
        None
    """
    save_folder = os.path.join(patient_data.study, "r2star_data")
    os.makedirs(save_folder, exist_ok=True)
    print(f"Saving R2* echo-time volumes to: {save_folder}")
    for idx, te in enumerate(patient_data.x):
        te_data = data_array[:, :, :, idx]
        nifti_filename = os.path.join(save_folder, f"te_{te:.2f}ms.nii.gz")
        save_nifti(nifti_filename, te_data, patient_data.affine)
