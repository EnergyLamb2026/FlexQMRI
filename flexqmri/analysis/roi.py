"""ROI-based analysis for qMRI parameter maps.

Extracts statistics from integer label maps and exports results to CSV.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.stats


def compute_roi_stats(
    param_map: np.ndarray,
    label_map: np.ndarray,
    labels: Optional[List[int]] = None,
) -> Dict[int, Dict[str, float]]:
    """Compute descriptive statistics for each ROI label.

    Args:
        param_map: 2D or 3D parameter map (e.g. f, D, R2*).
        label_map: Integer label map, same spatial shape as *param_map*.
        labels: Specific labels to analyse. If None, all non-zero labels are used.

    Returns:
        Dict mapping label → dict with 'mean', 'std', 'median', 'min', 'max', 'count'.
    """
    if labels is None:
        labels = sorted(np.unique(label_map[label_map > 0]).astype(int))

    stats: Dict[int, Dict[str, float]] = {}
    for label in labels:
        values = param_map[label_map == label]
        values = values[~np.isnan(values)]
        if len(values) == 0:
            continue
        stats[int(label)] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": int(len(values)),
        }
    return stats


def roi_histogram(
    param_map: np.ndarray,
    label_map: np.ndarray,
    label: int,
    bins: int = 50,
) -> tuple:
    """Compute histogram of parameter values within a single ROI.

    Args:
        param_map: Parameter map array.
        label_map: Integer label map.
        label: ROI label to extract.
        bins: Number of histogram bins.

    Returns:
        Tuple of (counts, bin_edges) from np.histogram.
    """
    values = param_map[label_map == label]
    values = values[~np.isnan(values)]
    return np.histogram(values, bins=bins)


def export_roi_csv(
    roi_results: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
    output_path: str,
) -> None:
    """Export ROI statistics to a single CSV file.

    Produces one row per (patient, map, label) combination.

    Args:
        roi_results: Nested dict ``{patient_id: {map_name: {label: stats_dict}}}``.
        output_path: Path to the output CSV file.
    """
    rows = []
    for patient_id, maps_dict in roi_results.items():
        for map_name, labels_dict in maps_dict.items():
            for label, stats in labels_dict.items():
                row = {
                    "patient_id": patient_id,
                    "map": map_name,
                    "label": label,
                }
                row.update(stats)
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def save_roi_stats(
    maps_name: str,
    maps: dict,
    mask: np.ndarray,
    map_keys: list,
    model_name: str,
    patient_id: str,
    study: str,
    output_dir: str,
    pipeline: str,
    confidence: float = 0.95,
    inference_time_per_voxel: float = None,
) -> None:
    """Compute masked ROI statistics for each parameter map and save them to CSV.

    Args:
        maps_name (str): Map type identifier (e.g. 'ivim', 'r2star').
        maps (dict): Dictionary mapping parameter names to 3-D numpy arrays.
        mask (np.ndarray): Boolean mask array (same spatial shape as each map).
        map_keys (list): Keys from ``maps`` to include in the output.
        model_name (str): Model identifier, used as the innermost folder level.
        patient_id (str): Patient identifier stored in the output table.
        study (str): Study identifier stored in the output table and used in the save path.
        output_dir (str): Root output directory.
        pipeline (str): Pipeline label (e.g. 'standard').
        confidence (float, optional): Confidence level for the CI. Defaults to 0.95.
        inference_time_per_voxel (float, optional): CPU time per voxel in seconds.
            When provided, an extra row with ``parameter='inference_time_s_per_voxel'``
            is appended. CI and n_voxels columns are NaN for that row. Defaults to None.

    Returns:
        None

    Raises:
        ValueError: If a key in ``map_keys`` is not present in ``maps``.

    Notes:
        The 95 % CI is computed as mean ± z * (std / sqrt(n)) where n is the
        number of unmasked voxels.  The CSV is written to
        ``{output_dir}/{patient_id}/{study}/{pipeline}/{maps_name}/{model_name}/stats.csv``.
    """
    z = scipy.stats.norm.ppf((1 + confidence) / 2.0)

    rows = []
    for key in map_keys:
        if key not in maps:
            raise ValueError(f"Key '{key}' not found in maps.")
        values = maps[key][mask.astype(bool)]
        n = values.size
        mean = float(np.mean(values))
        ci_length = z * float(np.std(values, ddof=1)) / np.sqrt(n)
        rows.append({
            "patient_id": patient_id,
            "study": study,
            "model": model_name,
            "parameter": key,
            "mean": mean,
            "low_CI95": mean - ci_length,
            "high_CI95": mean + ci_length,
            "n_voxels": n,
        })

    if inference_time_per_voxel is not None:
        rows.append({
            "patient_id": patient_id,
            "study": study,
            "model": model_name,
            "parameter": "inference_time_s_per_voxel",
            "mean": inference_time_per_voxel,
            "low_CI95": float("nan"),
            "high_CI95": float("nan"),
            "n_voxels": float("nan"),
        })

    save_dir = os.path.join(output_dir, patient_id, study, pipeline, maps_name, model_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "stats.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved ROI stats: {out_path}")
