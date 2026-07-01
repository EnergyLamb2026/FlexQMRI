"""Alignment utilities for resampling anatomical volumes onto a quantitative MRI grid.

Uses SimpleITK for spacing resampling and pad/crop. No registration.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from flexqmri.imaging.structures import PatientData


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def nifti_to_sitk(patient_data: PatientData) -> sitk.Image:
    """Convert PatientData to a SimpleITK image, preserving 4D structure.

    Args:
        patient_data: PatientData instance with .data, .pixel_spacing, .affine.

    Returns:
        SimpleITK image (3D scalar or 4D via JoinSeries).
    """
    data = patient_data.data

    if data.ndim == 3:
        data_t = np.transpose(data, (2, 1, 0))
        sitk_image = sitk.GetImageFromArray(data_t)
        sitk_image.SetSpacing(tuple(float(s) for s in patient_data.pixel_spacing[:3]))
        sitk_image.SetOrigin(tuple(float(o) for o in patient_data.affine[:3, 3]))
        sitk_image.SetDirection(list(patient_data.affine[:3, :3].flatten()))
        return sitk_image

    if data.ndim == 4:
        data_t = np.transpose(data, (2, 1, 0, 3))
        images_3d = []
        for t in range(data_t.shape[3]):
            img_3d = sitk.GetImageFromArray(data_t[:, :, :, t])
            img_3d.SetSpacing(tuple(float(s) for s in patient_data.pixel_spacing[:3]))
            img_3d.SetOrigin(tuple(float(o) for o in patient_data.affine[:3, 3]))
            img_3d.SetDirection(list(patient_data.affine[:3, :3].flatten()))
            images_3d.append(img_3d)
        return sitk.JoinSeries(images_3d)

    raise ValueError(f"Expected 3D or 4D data, got {data.ndim}D")


def copy_img_params(
    source_img: sitk.Image,
    target_array: np.ndarray,
    transpose: bool = True,
) -> sitk.Image:
    """Copy image parameters (spacing, origin, direction) from source to target.

    Args:
        source_img: SimpleITK image to copy parameters from.
        target_array: NumPy array to wrap in a SimpleITK image.
        transpose: If True, transpose (x,y,z) -> (z,y,x) for SimpleITK.

    Returns:
        SimpleITK image with matching metadata.
    """
    if transpose:
        target_array = np.transpose(target_array, (2, 1, 0))
    target_sitk = sitk.GetImageFromArray(target_array)

    src_dim = source_img.GetDimension()
    tgt_dim = target_sitk.GetDimension()

    if src_dim == 4 and tgt_dim == 3:
        target_sitk.SetSpacing(source_img.GetSpacing()[:3])
        target_sitk.SetOrigin(source_img.GetOrigin()[:3])
        target_sitk.SetDirection(source_img.GetDirection()[:9])
    elif src_dim == tgt_dim:
        target_sitk.SetSpacing(source_img.GetSpacing())
        target_sitk.SetOrigin(source_img.GetOrigin())
        target_sitk.SetDirection(source_img.GetDirection())
    else:
        raise ValueError(
            f"Dimension mismatch: source is {src_dim}D, target is {tgt_dim}D"
        )
    return target_sitk


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_to_spacing(
    image: sitk.Image,
    target_spacing: np.ndarray,
    interpolator: int = sitk.sitkLinear,
) -> sitk.Image:
    """Resample image to target spacing.

    Args:
        image: SimpleITK image.
        target_spacing: Desired voxel spacing (length 3).
        interpolator: SimpleITK interpolator constant. Defaults to linear.
            Use sitk.sitkNearestNeighbor for label/mask data.

    Returns:
        Resampled SimpleITK image.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = tuple(
        max(1, int(os * original_spacing[i] / target_spacing[i]))
        for i, os in enumerate(original_size)
    )
    resample = sitk.ResampleImageFilter()
    resample.SetSize(new_size)
    resample.SetOutputSpacing(tuple(float(s) for s in target_spacing))
    resample.SetInterpolator(interpolator)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    return resample.Execute(image)


# ---------------------------------------------------------------------------
# Pad / crop alignment
# ---------------------------------------------------------------------------

def compute_pad_crop_params(
    fixed: sitk.Image, moving: sitk.Image
) -> Dict[str, List[int]]:
    """Compute pad/crop parameters to align *moving* to *fixed*.

    Args:
        fixed: Reference SimpleITK image.
        moving: Moving SimpleITK image.

    Returns:
        Dict with 'pad_before', 'pad_after', 'crop_before', 'crop_after' lists.
    """
    params: Dict[str, List[int]] = {
        "pad_before": [],
        "pad_after": [],
        "crop_before": [],
        "crop_after": [],
    }
    for dim in range(3):
        diff = fixed.GetSize()[dim] - moving.GetSize()[dim]
        if diff > 0:
            params["pad_before"].append(int(np.ceil(diff / 2)))
            params["pad_after"].append(int(np.floor(diff / 2)))
            params["crop_before"].append(0)
            params["crop_after"].append(0)
        elif diff < 0:
            params["pad_before"].append(0)
            params["pad_after"].append(0)
            params["crop_before"].append(int(np.ceil(-diff / 2)))
            params["crop_after"].append(int(np.floor(-diff / 2)))
        else:
            for key in params:
                params[key].append(0)
    return params


def pad_dimension(image: sitk.Image, dim: int, before: int, after: int) -> sitk.Image:
    """Pad a single dimension of an image.

    Args:
        image: SimpleITK image.
        dim: Dimension index (0, 1, or 2).
        before: Voxels to add before.
        after: Voxels to add after.

    Returns:
        Padded image.
    """
    pads_before = [abs(int(before)) if i == dim else 0 for i in range(3)]
    pads_after = [abs(int(after)) if i == dim else 0 for i in range(3)]
    pad_filter = sitk.ConstantPadImageFilter()
    pad_filter.SetPadLowerBound(pads_before)
    pad_filter.SetPadUpperBound(pads_after)
    return pad_filter.Execute(image)


def crop_dimension(image: sitk.Image, dim: int, before: int, after: int) -> sitk.Image:
    """Crop a single dimension of an image.

    Args:
        image: SimpleITK image.
        dim: Dimension index (0, 1, or 2).
        before: Voxels to remove from start.
        after: Voxels to remove from end.

    Returns:
        Cropped image.
    """
    crops_before = [abs(int(before)) if i == dim else 0 for i in range(3)]
    crops_after = [abs(int(after)) if i == dim else 0 for i in range(3)]
    crop_filter = sitk.CropImageFilter()
    crop_filter.SetLowerBoundaryCropSize(crops_before)
    crop_filter.SetUpperBoundaryCropSize(crops_after)
    return crop_filter.Execute(image)


def apply_pad_crop_params(
    image: sitk.Image, params: Dict[str, List[int]]
) -> sitk.Image:
    """Apply precomputed pad/crop parameters to an image.

    Args:
        image: SimpleITK image to transform.
        params: Dict from :func:`compute_pad_crop_params`.

    Returns:
        Transformed image.
    """
    result = image
    for dim in range(3):
        result = pad_dimension(result, dim, params["pad_before"][dim], params["pad_after"][dim])
        result = crop_dimension(result, dim, params["crop_before"][dim], params["crop_after"][dim])
    return result


def get_aligned_images_pad_crop(
    fixed: sitk.Image,
    moving: sitk.Image,
    compute_params: bool = False,
) -> Tuple[sitk.Image, sitk.Image, Optional[Dict[str, List[int]]]]:
    """Align *moving* to *fixed* via pad/crop.

    Args:
        fixed: Reference image (not modified).
        moving: Image to align.
        compute_params: If True, compute and return the parameters.

    Returns:
        (fixed, moving_aligned, params). params is None when compute_params is False.
    """
    params = compute_pad_crop_params(fixed, moving) if compute_params else None
    moving_aligned = apply_pad_crop_params(moving, params)
    return fixed, moving_aligned, params


# ---------------------------------------------------------------------------
# 4D helpers
# ---------------------------------------------------------------------------

def extract_3d_timepoint(sitk_4d: sitk.Image, t_index: int) -> sitk.Image:
    """Extract a single 3D volume from a 4D SimpleITK image.

    Args:
        sitk_4d: 4D SimpleITK image.
        t_index: Temporal index.

    Returns:
        3D SimpleITK image.
    """
    filt = sitk.ExtractImageFilter()
    filt.SetSize([sitk_4d.GetSize()[0], sitk_4d.GetSize()[1], sitk_4d.GetSize()[2], 0])
    filt.SetIndex([0, 0, 0, t_index])
    return filt.Execute(sitk_4d)


def align_patient_data(
    quantitative_data: PatientData,
    anatomical_data: PatientData,
) -> Dict[str, Optional[np.ndarray]]:
    """Resample anatomical data (and its mask) onto the quantitative grid.

    The quantitative volume is left untouched so that fitting runs at
    native resolution without interpolation artifacts. The anatomical
    image and its mask are resampled to the quantitative spacing, then
    pad/cropped to match the quantitative array shape.

    Args:
        quantitative_data: PatientData with a 3D or 4D image whose grid
            (spacing and shape) defines the output. For 4D inputs, the
            first volume defines the grid.
        anatomical_data: PatientData with a 3D anatomical image. The mask,
            if present, is taken from this PatientData and resampled with
            nearest-neighbor interpolation.

    Returns:
        Dict with:
            'aligned_anatomical': np.ndarray, anatomical resampled to the
                quantitative grid (3D, shape matching quantitative t=0).
            'aligned_mask': np.ndarray or None, anatomical mask resampled
                to the quantitative grid (3D), or None if no mask.

    Notes:
        Pad/crop is applied symmetrically in array index space; the result
        is shape-aligned to the quantitative grid but is not a physical
        registration (no rigid/affine transform is estimated).
    """
    quant_sitk = nifti_to_sitk(quantitative_data)
    anat_sitk = nifti_to_sitk(anatomical_data)

    quant_3d = (
        extract_3d_timepoint(quant_sitk, 0)
        if quantitative_data.data.ndim == 4
        else quant_sitk
    )

    resampled_anat = resample_to_spacing(anat_sitk, quantitative_data.pixel_spacing)
    _, aligned_anat, params = get_aligned_images_pad_crop(
        quant_3d, resampled_anat, compute_params=True
    )
    aligned_anat_array = np.transpose(sitk.GetArrayFromImage(aligned_anat), (2, 1, 0))

    aligned_mask_array = None
    if anatomical_data.has_mask():
        mask_sitk = copy_img_params(anat_sitk, anatomical_data.mask_array)
        resampled_mask = resample_to_spacing(
            mask_sitk,
            quantitative_data.pixel_spacing,
            interpolator=sitk.sitkNearestNeighbor,
        )
        aligned_mask = apply_pad_crop_params(resampled_mask, params)
        aligned_mask_array = np.transpose(
            sitk.GetArrayFromImage(aligned_mask), (2, 1, 0)
        )

    return {
        "aligned_anatomical": aligned_anat_array,
        "aligned_mask": aligned_mask_array,
    }
