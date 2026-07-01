"""Functions to load patient MRI data (DICOM/NIfTI) for qMRI fitting."""
import os

import nibabel
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from totalsegmentator.python_api import totalsegmentator

import flexqmri.imaging.filters as filter
from flexqmri.convert.dicom2nifti import convert_series_dicom_to_nifti, convert_siemens_dicom_to_nifti
from flexqmri.imaging.structures import PatientData

# =============================================================================
# Helper functions
# =============================================================================

_NIFTI_EXTENSIONS = ('.nii', '.nii.gz')


def _ensure_series_nifti(series_folder: str) -> None:
    """Convert a series DICOM folder to NIfTI if the images/ subdirectory has no NIfTI files.

    The function is a no-op when NIfTI files already exist in
    ``<series_folder>/images/``.  Otherwise it runs ``dicom2nifti`` on the
    series folder and writes the output into the ``images/`` subdirectory.

    Args:
        series_folder (str): Path to a single series directory.  DICOM files
            are expected directly inside this directory.

    Returns:
        None

    Raises:
        RuntimeError: If the conversion fails (propagated from
            :func:`~flexqmri.convert.dicom2nifti.convert_series_dicom_to_nifti`).
    """
    images_folder = os.path.join(series_folder, 'images')
    if os.path.isdir(images_folder):
        if any(f.endswith(_NIFTI_EXTENSIONS) for f in os.listdir(images_folder)):
            return
    print(f"No NIfTI files found in '{images_folder}'. Converting DICOM from '{series_folder}'...")
    convert_series_dicom_to_nifti(series_folder, images_folder)


def _find_or_convert_nifti(data_folder: str, series: str) -> str:
    """Return the path to a NIfTI file in *data_folder*, converting DICOM if needed.

    Searches *data_folder* for a file with a `.nii` or `.nii.gz` extension and
    returns its absolute path.  When no NIfTI file is found, or when the
    ``images/`` directory does not exist, the function treats the parent series
    folder as a DICOM source, converts it, and returns the path of the first
    resulting NIfTI file.

    Args:
        data_folder (str): Path to the ``images/`` directory for the series.
        series (str): Series name used only for error messages.

    Returns:
        str: Absolute path to a NIfTI file ready for loading.

    Raises:
        ValueError: If no NIfTI files are found after conversion.
    """
    if not os.path.isdir(data_folder):
        series_folder = os.path.dirname(data_folder)
        convert_series_dicom_to_nifti(series_folder, data_folder)

    nifti_files = [f for f in os.listdir(data_folder) if f.endswith(_NIFTI_EXTENSIONS)]
    if nifti_files:
        return os.path.join(data_folder, nifti_files[0])

    # images/ exists but contains only DICOM files — convert from parent
    series_folder = os.path.dirname(data_folder)
    print(f"No NIfTI in '{data_folder}'. Converting DICOM from '{series_folder}'...")
    convert_series_dicom_to_nifti(series_folder, data_folder)

    nifti_files = [f for f in os.listdir(data_folder) if f.endswith(_NIFTI_EXTENSIONS)]
    if not nifti_files:
        raise ValueError(
            f"Conversion produced no NIfTI files in '{data_folder}' for series '{series}'."
        )
    return os.path.join(data_folder, nifti_files[0])


def get_pixel_spacing_from_image(nifti_image) -> np.ndarray:
    """
    Extract pixel/voxel spacing from a NIfTI image.
    
    Parameters
    ----------
    nifti_image : nibabel.Nifti1Image
        NIfTI image object
    
    Returns
    -------
    np.ndarray
        Pixel spacing in mm for each dimension (shape: (ndim,))
    """
    if nifti_image is None:
        return None
    
    # Get the header
    header = nifti_image.header
    
    # Get zooms (pixel spacing) from header
    # zooms contains the voxel sizes in mm for each dimension
    zooms = np.array(header.get_zooms())
    
    # Usually first 3 dimensions are spatial (x, y, z)
    # Return only the spatial dimensions (first 3)
    if len(zooms) >= 3:
        return zooms[:3]
    else:
        return zooms


# =============================================================================
# Factory functions to load data into PatientData
# =============================================================================

def prepare_study(config: dict, patient_id: str, study: str) -> tuple:
    """Return the study path and its series list, converting from DICOM first if needed.

    Args:
        config (dict): Configuration dictionary with keys ``paths.nifti_data``
            and ``paths.dicom_data``.
        patient_id (str): Patient identifier.
        study (str): Study folder name.

    Returns:
        tuple: (study_path, all_series) — resolved path and sorted series list.
    """
    nifti_root = config['paths']['nifti_data']
    study_path = os.path.join(nifti_root, patient_id, study)

    if not os.path.isdir(study_path) or not os.listdir(study_path):
        print(f"NIfTI study not found at '{study_path}'. Running DICOM conversion...")
        convert_siemens_dicom_to_nifti(config['paths']['dicom_data'], nifti_root, patient_id)

    return study_path, sorted(os.listdir(study_path))


def _load_offset_patient_data(
    study_path: str,
    all_series: list,
    patient_id: str,
    keywords: list,
    include: list,
    exclude: list,
    offset_path: str,
    serie_name: str,
    slice_idx: int = None,
) -> PatientData:
    """Load multi-offset NIfTI data into a PatientData object.

    Shared implementation used by the IVIM and T2* patient loaders. Filters
    series, auto-converts DICOM if needed, loads all offset volumes, and
    assembles a :class:`PatientData`.

    Args:
        study_path (str): Path to the study directory.
        all_series (list): List of all series present in the study.
        patient_id (str): Patient identifier.
        keywords (list): Keywords that must all be present in a series name.
        include (list): Patterns where at least one must appear in the series name.
        exclude (list): Patterns that must not appear in the series name.
        offset_path (str): Path to the frequency offset file.
        serie_name (str): Label stored on the returned :class:`PatientData`.
        slice_idx (int, optional): Slice index to extract. Keeps all slices
            when ``None``. Defaults to ``None``.

    Returns:
        PatientData: Loaded data object.

    Raises:
        ValueError: If no series match the provided filters.
    """
    matched_series = filter.filter_series(
        all_series,
        keywords=keywords,
        include=include,
        exclude=exclude,
    )

    if not matched_series:
        raise ValueError(f"No {serie_name} series found with keywords {keywords}")

    for serie in matched_series:
        _ensure_series_nifti(os.path.join(study_path, serie))

    image_list = load_all_offsets_nifti(study_path, matched_series)
    data_array = img_to_offset_array(image_list)
    affine = image_list[0].affine if image_list else None
    pixel_spacing = get_pixel_spacing_from_image(image_list[0]) if image_list else None

    return PatientData(
        data=data_array,
        patient_id=patient_id,
        study=study_path,
        serie_name=serie_name,
        original_shape=data_array.shape,
        offset_path=offset_path,
        mask_array=None,
        affine=affine,
        pixel_spacing=pixel_spacing,
        slice_idx=slice_idx,
    )


def segment_with_totalseg(image: nibabel.Nifti1Image, structure: str) -> np.ndarray:
    """Run TotalSegmentator on a NIfTI image and return a binary mask.

    For ``'total_mr'`` all segmented classes are binarized to 1.  For single
    organ structures (``'pancreas'``, ``'liver'``) only the requested organ is
    segmented via ``roi_subset``, which is faster.

    Args:
        image (nibabel.Nifti1Image): Input NIfTI image to segment.
        structure (str): Target structure.  Supported values are listed in
            the inference config under ``anatomical.mask.structure``.

    Returns:
        np.ndarray: Binary mask with the same spatial shape as *image*.
    """
    print("Beginning segmentation with TotalSegmentator...")
    seg = totalsegmentator(image, task='total_mr')
    seg_data = seg.get_fdata().astype(np.uint8)

    if structure == 'pancreas':
        mask = (seg_data == 7).astype(np.uint8)
    elif structure == 'liver':
        mask = (seg_data == 5).astype(np.uint8)
    elif structure == 'total_mr':
        mask = (seg_data != 0).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported structure for TotalSegmentator: {structure}")

    return mask


def load_general_patient_data(
    study_path: str,
    all_series: list,
    config: dict,
    patient_id: str,
    keywords: list,
    include: list,
    exclude: list,
    slice_idx: int = None,
) -> PatientData:
    """Load arbitrary single-volume MRI data. Can run TotalSegmentator to generate a mask.

    Filters *all_series* by *keywords*, *include*, and *exclude*, then loads
    the first matching series as a 3-D NIfTI volume.  The mask is derived from
    TotalSegmentator using the structure configured under
    ``config['anatomical']['mask']['structure']``.

    Args:
        study_path (str): Path to the study directory.
        all_series (list): List of all series present in the study.
        config (dict): Configuration dictionary.  The function reads
            ``config['anatomical']['mask']`` for mask settings.
        patient_id (str): Patient identifier.
        keywords (list): Keywords that must all be present in the series name.
        include (list): Patterns where at least one must appear in the series
            name.
        exclude (list): Patterns that must not appear in the series name.
        slice_idx (int, optional): Slice index to extract.  Keeps all slices
            when ``None``.  Defaults to ``None``.

    Returns:
        PatientData: Object containing image data, TotalSegmentator mask,
            affine, and metadata.

    Raises:
        ValueError: If no series match the provided filters.
        ValueError: If the configured structure is not supported.

    Notes:
        TotalSegmentator must be installed (``pip install TotalSegmentator``).
        Supported structures are 'pancreas', 'liver', and 'total_mr'.
        For 'total_mr' all segmented classes are collapsed to a binary mask.
    """
    series = filter.filter_series(
        all_series,
        keywords=keywords,
        include=include,
        exclude=exclude,
        order=False,
    )

    if not series:
        raise ValueError(
            f"No series found matching keywords={keywords}, include={include}, exclude={exclude}"
        )

    _ensure_series_nifti(os.path.join(study_path, series[0]))
    image_array, affine, image = load_nifti_data(study_path, series[0])

    mask_config = config['anatomical']['mask']
    method = mask_config['method']
    if method is None:
        mask_array = np.ones(image_array.shape, dtype=np.uint8)
    elif method == 'totalsegmentator':
        mask_path = os.path.join(study_path, series[0], 'mask', f"{mask_config['structure']}.nii.gz")
        if os.path.exists(mask_path):
            print(f"Loading cached mask from: {os.path.abspath(mask_path)}")
            mask_array, _, _ = load_nifti(mask_path, return_img=True)
        else:
            mask_array = segment_with_totalseg(image, mask_config['structure'])
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            print(f"Saving mask to: {os.path.abspath(mask_path)}")
            save_nifti(mask_path, mask_array, affine)
    elif method == 'nifti':
        mask_array, _, _ = load_nifti(mask_config['path'], return_img=True)
    else:
        raise ValueError(f"Invalid mask method '{method}'. Options: null, 'totalsegmentator', 'nifti'.")

    pixel_spacing = get_pixel_spacing_from_image(image)

    return PatientData(
        data=image_array,
        patient_id=patient_id,
        study=study_path,
        serie_name='_'.join(keywords),
        original_shape=image_array.shape,
        mask_array=mask_array,
        affine=affine,
        pixel_spacing=pixel_spacing,
        slice_idx=slice_idx,
    )


# =============================================================================
# Core loading functions
# =============================================================================


def load_nifti_data(study_path, series: str) -> tuple:
    """Load a NIfTI file for a series and return data, image and affine.

    The function looks for a `.nii` or `.nii.gz` file inside
    ``<study_path>/<series>/images/``.  If no NIfTI file is found, the folder
    is treated as a DICOM series and automatically converted to NIfTI before
    loading (see :func:`_find_or_convert_nifti`).

    Args:
        study_path (str): Path to the study directory.
        series (str): Series name.  The function reads from
            ``<study_path>/<series>/images/``.

    Returns:
        tuple:
            - **data** (*np.ndarray*): Image data array.
            - **affine** (*np.ndarray*): Affine transformation matrix.
            - **image** (*nibabel.Nifti1Image*): NIfTI image object.

    Raises:
        ValueError: If the ``images/`` directory is empty or contains no
            recognisable files.

    Notes:
        Auto-conversion from DICOM requires SimpleITK to be installed.
        The converted NIfTI file is written into the same ``images/``
        directory alongside the original DICOM files.
    """
    data_folder = os.path.join(study_path, series, 'images')
    image_path = _find_or_convert_nifti(data_folder, series)

    data, affine, image = load_nifti(image_path, return_img=True)

    return data, affine, image

def load_all_offsets_nifti(study_path: str, series: list) -> list:
    """Load one NIfTI image per offset series from disk.

    Args:
        study_path (str): Path to the study directory.
        series (list): List of series names, one per frequency offset.

    Returns:
        list: List of nibabel.nifti1.Nifti1Image, one per offset.
    """
    images = []
    for offset in series:
        _, _, image = load_nifti_data(study_path, offset)
        images.append(image)
    return images

def img_to_offset_array(images): 
    """
    Convert a list of nibabel.nifti1.Nifti1Image to a numpy array.

    Parameters:
        images (list): list of nibabel.nifti1.Nifti1Image
    
    Returns:
        np.ndarray: 4D numpy array
    """
    offset_array = np.zeros(images[0].get_fdata().shape + (len(images),))
    for i in range(len(images)):
        offset_array[:, :, :, i] = images[i].get_fdata()

    return offset_array

