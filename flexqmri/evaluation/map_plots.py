"""Plotting of qMRI parameter maps on a single slice."""

import os

import numpy as np
import matplotlib.pyplot as plt


def plot_maps_slice(
    param_maps: dict,
    map_keys: list,
    titles: list,
    modality: str,
    model_name: str,
    output_dir: str,
    patient_id: str,
    study: str,
    mask_array: np.ndarray = None,
    anatomical: np.ndarray = None,
    slice_idx: int = None,
    cmap=None,
    colorbar_labels: list = None,
    save: bool = True,
    display: bool = False,
    remove_outliers: bool = False,
) -> None:
    """Plot a single slice of parameter maps for a given MRI modality.

    When an anatomical volume is provided, it is shown as a greyscale background
    and the parameter maps are overlaid with a colormap and alpha. If a mask is
    provided, the overlay is restricted to masked voxels.

    Args:
        param_maps (dict): Maps from key name to ndarray of shape (x, y, z).
        map_keys (list): Keys into param_maps to plot, one subplot each.
        titles (list): Display title for each subplot, aligned with map_keys.
        modality (str): Modality label (e.g. 'ivim', 'r2star'), used in the title
            and filename.
        model_name (str): Name of the model used for fitting, added to the title and filename.
        output_dir (str): Root output directory (config['paths']['output_data']).
        patient_id (str): Patient identifier, used in the save path and title.
        study (str): Study name, used in the save path.
        mask_array (np.ndarray, optional): 3D binary mask. If provided, the colormap
            overlay is restricted to masked voxels.
        anatomical (np.ndarray, optional): 3D anatomical volume on the same grid as the
            parameter maps, used as greyscale background. If None, maps are shown alone.
        slice_idx (int, optional): Slice index along z for display and filename. If the
            volume has only one slice (pre-cropped), the array is indexed at 0 regardless.
            Defaults to the middle slice of the volume.
        cmap (str or Colormap, optional): Colormap for the overlay. Defaults to 'viridis'.
        colorbar_labels (list, optional): Colorbar axis label for each subplot, aligned
            with map_keys (e.g. unit strings like 'mm²/s' or 's⁻¹'). Defaults to None.
        save (bool): Save the figure to disk. Defaults to True.
        display (bool): Call plt.show(). Defaults to False.
        remove_outliers (bool): Remove outliers using IQR method for colormap scaling. Defaults to False.

    Returns:
        None

    Notes:
        Saves to output_dir/patient_id/study/{model_name}_{modality}_maps_slice{slice_idx}.png.
        When remove_outliers=True, color limits use the IQR method on the 5th/95th percentiles.
    """
    if cmap is None:
        cmap = 'viridis'

    n_slices = next(iter(param_maps.values())).shape[2]
    if n_slices == 1:
        array_idx = 0
    elif slice_idx is not None:
        array_idx = slice_idx
    elif mask_array is not None:
        array_idx = int(np.argmax(mask_array.sum(axis=(0, 1))))
    else:
        array_idx = n_slices // 2
    if slice_idx is None:
        slice_idx = array_idx

    anat_slice = anatomical[:, :, array_idx].T if anatomical is not None else None
    mask_slice = mask_array[:, :, array_idx].T if mask_array is not None else None

    fig, axes = plt.subplots(1, len(map_keys), figsize=(4 * len(map_keys), 4))
    if len(map_keys) == 1:
        axes = [axes]
    fig.suptitle(f"{modality} maps — {patient_id}, {model_name}, slice {slice_idx}", fontsize=13)

    if colorbar_labels is None:
        colorbar_labels = [None] * len(map_keys)

    for ax, key, title, colorbar_label in zip(axes, map_keys, titles, colorbar_labels):
        data_slice = param_maps[key][:, :, array_idx].T

        # Compute color limits if auto-scaling and remove_outliers is True
        if remove_outliers:
            data_flat = param_maps[key].flatten()
            data_flat = data_flat[~np.isnan(data_flat)]
            if len(data_flat) > 0:
                q1, q3 = np.percentile(data_flat, [5, 95])
                iqr = q3 - q1
                vmin_slice = q1 - 1.5 * iqr
                vmax_slice = q3 + 1.5 * iqr
        else:
            vmin_slice, vmax_slice = None, None

        if anat_slice is not None:
            ax.imshow(anat_slice, cmap='gray', aspect='equal')
            overlay = np.ma.masked_where(mask_slice == 0, data_slice) if mask_slice is not None else data_slice
            im = ax.imshow(overlay, cmap=cmap, aspect='equal', alpha=0.6,
                           vmin=vmin_slice, vmax=vmax_slice)
        else:
            im = ax.imshow(data_slice, cmap=cmap, aspect='equal',
                           vmin=vmin_slice, vmax=vmax_slice)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if colorbar_label is not None:
            cb.set_label(colorbar_label, fontsize=9)

    plt.tight_layout()

    if save:
        save_dir = os.path.join(output_dir, patient_id, study)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{model_name}_{modality}_maps_slice{slice_idx}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if display:
        plt.show()

    plt.close(fig)
