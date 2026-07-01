"""Model comparison figure and clinical summary table across IVIM and R2* parameters."""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import flexqmri.imaging.config

from flexqmri.analysis.repeatability import compute_icc, compute_inter_patient_stats, load_stats
from flexqmri.evaluation.repeatability_plots import plot_model_boxplots, plot_model_comparison_with_icc

_MODEL_LABELS: dict[str, str] = {
    "vendor": "LSQ (manufacturer)",
    "trf": "LSQ (TRF)",
    "mlp": "MLP",
    "transformer": "Transformer",
    "ncde": "NCDE",
}

_PARAM_LABELS: dict[str, str] = {
    "f": "f",
    "D": "D (mm²/s)",
    "D_star": "D* (mm²/s)",
    "R2star": "R2* (s⁻¹)",
}

_IVIM_PARAMS: list[str] = ["f", "D", "D_star"]
_R2STAR_PARAMS: list[str] = ["R2star"]

# Explicit y-axis ticks for the Value row, per parameter, in raw data units.
# A ×10^n factor is appended only for small magnitudes (tick exponent <= -2, e.g. D);
# larger values keep their raw labels. Comment out an entry to fall back to auto ticks.
_PARAM_YTICKS: dict[str, list[float]] = {
    "f": [0.15, 0.3, 0.45],
    "D": [1.0e-3, 2.0e-3, 3.0e-3],
    "D_star": [0.08, 0.11, 0.14],
    "R2star": [10, 90, 180],
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Model comparison figure and clinical table across IVIM and R2* parameters."
    )
    parser.add_argument("-p", "--patient_ids", type=str, nargs="+", required=True,
                        help="List of patient identifiers.")
    parser.add_argument("-s1", "--study1", type=str, default="Study1",
                        help="First session folder name.")
    parser.add_argument("-s2", "--study2", type=str, default="Study2",
                        help="Second session folder name.")
    parser.add_argument("--models", type=str, nargs="+", default=["trf", "mlp", "transformer", "ncde"],
                        help="Model identifiers in display order (default: trf mlp transformer ncde).")
    parser.add_argument("--ivim_maps_name", type=str, default="ivim",
                        help="maps_name folder for IVIM parameters (default: 'ivim').")
    parser.add_argument("--r2star_maps_name", type=str, default="r2star",
                        help="maps_name folder for R2star (default: 'r2star').")
    parser.add_argument("--pipeline", type=str, default="standard",
                        help="Pipeline label (default: 'standard').")
    parser.add_argument("-c", "--common_config_path", type=str, required=True,
                        help="Path to the common config file.")
    return parser.parse_args()


def _collect_sessions(
    output_dir: str,
    patient_ids: list,
    study1: str,
    study2: str,
    pipeline: str,
    model_name: str,
    maps_name: str,
) -> tuple:
    """Load per-patient session means for all parameters of one model.

    Args:
        output_dir (str): Root output directory.
        patient_ids (list): Patient identifiers.
        study1 (str): First session folder name.
        study2 (str): Second session folder name.
        pipeline (str): Pipeline label.
        model_name (str): Model identifier.
        maps_name (str): Map type identifier.

    Returns:
        tuple: (session1, session2) each a defaultdict mapping parameter name
            to a list of per-patient means aligned by index. Missing patients
            are skipped with a warning.
    """
    session1: dict = defaultdict(list)
    session2: dict = defaultdict(list)
    for pid in patient_ids:
        try:
            df1 = load_stats(output_dir, pid, study1, pipeline, model_name, maps_name)
            df2 = load_stats(output_dir, pid, study2, pipeline, model_name, maps_name)
        except FileNotFoundError as e:
            print(f"WARNING: {e}, skipping.")
            continue
        for _, row1 in df1.iterrows():
            param = row1["parameter"]
            match = df2[df2["parameter"] == param]
            if match.empty:
                print(f"WARNING: '{param}' not found for {pid}/{study2}, skipping.")
                continue
            session1[param].append(float(row1["mean"]))
            session2[param].append(float(match["mean"].iloc[0]))
    return session1, session2


def main():
    args = parse_args()
    config = flexqmri.imaging.config.load_config_hierarchical("ivim", args.common_config_path)
    output_dir = config["paths"]["output_data"]

    figure_data: dict = {}
    table_rows = []

    for model_name in args.models:
        s1_ivim, s2_ivim = _collect_sessions(
            output_dir, args.patient_ids, args.study1, args.study2,
            args.pipeline, model_name, args.ivim_maps_name,
        )
        for param in _IVIM_PARAMS:
            figure_data.setdefault(param, {})[model_name] = {
                "S1": list(s1_ivim.get(param, [])),
                "S2": list(s2_ivim.get(param, [])),
            }
            s1 = np.array(s1_ivim.get(param, []))
            s2 = np.array(s2_ivim.get(param, []))
            if len(s1) >= 2:
                stats = compute_inter_patient_stats((s1 + s2) / 2)
                table_rows.append({
                    "model": _MODEL_LABELS.get(model_name, model_name),
                    "parameter": param,
                    "n_patients": len(s1),
                    "mean": stats["mean"],
                    "ci_lower": stats["ci_lower"],
                    "ci_upper": stats["ci_upper"],
                })

        s1_r2star, s2_r2star = _collect_sessions(
            output_dir, args.patient_ids, args.study1, args.study2,
            args.pipeline, model_name, args.r2star_maps_name,
        )
        for param in _R2STAR_PARAMS:
            figure_data.setdefault(param, {})[model_name] = {
                "S1": list(s1_r2star.get(param, [])),
                "S2": list(s2_r2star.get(param, [])),
            }
            s1 = np.array(s1_r2star.get(param, []))
            s2 = np.array(s2_r2star.get(param, []))
            if len(s1) >= 2:
                stats = compute_inter_patient_stats((s1 + s2) / 2)
                table_rows.append({
                    "model": _MODEL_LABELS.get(model_name, model_name),
                    "parameter": param,
                    "n_patients": len(s1),
                    "mean": stats["mean"],
                    "ci_lower": stats["ci_lower"],
                    "ci_upper": stats["ci_upper"],
                })

    s1_vendor, s2_vendor = _collect_sessions(
        output_dir, args.patient_ids, args.study1, args.study2,
        args.pipeline, "vendor", args.r2star_maps_name,
    )
    for param in _R2STAR_PARAMS:
        figure_data.setdefault(param, {})["vendor"] = {
            "S1": list(s1_vendor.get(param, [])),
            "S2": list(s2_vendor.get(param, [])),
        }
        s1 = np.array(s1_vendor.get(param, []))
        s2 = np.array(s2_vendor.get(param, []))
        if len(s1) >= 2:
            stats = compute_inter_patient_stats((s1 + s2) / 2)
            table_rows.append({
                "model": _MODEL_LABELS["vendor"],
                "parameter": param,
                "n_patients": len(s1),
                "mean": stats["mean"],
                "ci_lower": stats["ci_lower"],
                "ci_upper": stats["ci_upper"],
            })

    save_dir = os.path.join(output_dir, "analysis", args.pipeline)
    os.makedirs(save_dir, exist_ok=True)

    table_df = pd.DataFrame(table_rows)
    print(table_df.to_string(index=False, float_format="{:.6f}".format))
    table_df.to_csv(os.path.join(save_dir, "model_comparison.csv"), index=False)
    print(f"\nSaved clinical table: {save_dir}/model_comparison.csv")

    r2star_model_order = ["vendor"] + args.models
    param_model_order = {p: args.models for p in _IVIM_PARAMS}
    param_model_order.update({p: r2star_model_order for p in _R2STAR_PARAMS})

    icc_data: dict = {}
    for param in _IVIM_PARAMS + _R2STAR_PARAMS:
        icc_data[param] = {}
        for model_name in param_model_order[param]:
            entry = figure_data.get(param, {}).get(model_name, {})
            s1 = np.array(entry.get("S1", []))
            s2 = np.array(entry.get("S2", []))
            if len(s1) >= 3 and len(s1) == len(s2):
                icc_data[param][model_name] = compute_icc(s1, s2)

    plot_model_boxplots(
        data=figure_data,
        param_order=_IVIM_PARAMS + _R2STAR_PARAMS,
        param_labels=_PARAM_LABELS,
        model_order=param_model_order,
        model_labels=_MODEL_LABELS,
        suptitle=f"Model comparison — {args.study1} vs {args.study2}",
        save_path=os.path.join(save_dir, "model_comparison.png"),
        param_yticks=_PARAM_YTICKS,
    )

    plot_model_comparison_with_icc(
        data=figure_data,
        icc_data=icc_data,
        param_order=_IVIM_PARAMS + _R2STAR_PARAMS,
        param_labels=_PARAM_LABELS,
        model_order=param_model_order,
        model_labels=_MODEL_LABELS,
        suptitle=f"Model comparison — {args.study1} vs {args.study2}",
        save_path=os.path.join(save_dir, "model_comparison_icc_bar.png"),
        icc_style="bar",
        param_yticks=_PARAM_YTICKS,
    )


if __name__ == "__main__":
    main()
