"""Intra-patient repeatability analysis: ICC(2,1) across patients with two sessions."""

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import flexqmri.imaging.config

from flexqmri.analysis.repeatability import compute_icc_pingouin, load_stats
from flexqmri.evaluation.repeatability_plots import plot_paired_sessions, plot_pooled_sessions, plot_spearman_matrix

# Patient pipeline script to run per session for a given map type.
_PATIENT_SCRIPTS = {"ivim": "ivim_patient.py", "r2star": "r2star_patient.py"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Intra-patient repeatability analysis via ICC.")
    parser.add_argument("-p", "--patient_ids", type=str, nargs="+", required=True,
                        help="List of patient identifiers.")
    parser.add_argument("-s1", "--study1", type=str, default="Study1",
                        help="First session folder name inside the patient NIfTI directory.")
    parser.add_argument("-s2", "--study2", type=str, default="Study2",
                        help="Second session folder name inside the patient NIfTI directory.")
    parser.add_argument("--maps_name", type=str, required=True,
                        help="Map type identifier ('ivim' or 'r2star').")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model identifier used as the innermost folder level.")
    parser.add_argument("--pipeline", type=str, default="standard",
                        help="Pipeline label used to locate outputs (default: 'standard').")
    parser.add_argument("-c", "--common_config_path", type=str, required=True,
                        help="Path to the common config file.")
    parser.add_argument("--run_pipeline", action="store_true",
                        help="Run the patient pipeline (ivim_patient.py or r2star_patient.py) "
                             "for each patient and session before analyzing.")
    return parser.parse_args()


def run_patient_pipeline(maps_name: str, patient_id: str, study: str,
                         common_config_path: str, pipeline: str) -> None:
    """Run the patient pipeline script for a single session via subprocess.

    Args:
        maps_name (str): Map type ('ivim' or 'r2star'); selects the script to run.
        patient_id (str): Patient identifier.
        study (str): Session folder name.
        common_config_path (str): Path to the common config file.
        pipeline (str): Pipeline label forwarded to the patient script.

    Returns:
        None

    Raises:
        KeyError: If ``maps_name`` is not a supported map type.
        subprocess.CalledProcessError: If the patient script exits with a non-zero code.
    """
    script = Path(__file__).parent / _PATIENT_SCRIPTS[maps_name]
    cmd = [
        sys.executable, str(script),
        "-p", patient_id, "-s", study, "-c", common_config_path,
        "--pipeline", pipeline,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    config = flexqmri.imaging.config.load_config_hierarchical("ivim", args.common_config_path)
    output_dir = config["paths"]["output_data"]

    if args.run_pipeline:
        for pid in args.patient_ids:
            run_patient_pipeline(args.maps_name, pid, args.study1, args.common_config_path, args.pipeline)
            run_patient_pipeline(args.maps_name, pid, args.study2, args.common_config_path, args.pipeline)

    session1: dict[str, list[float]] = defaultdict(list)
    session2: dict[str, list[float]] = defaultdict(list)
    patient_ids_per_param: dict[str, list[str]] = defaultdict(list)

    for pid in args.patient_ids:
        df1 = load_stats(output_dir, pid, args.study1, args.pipeline, args.model_name, args.maps_name)
        df2 = load_stats(output_dir, pid, args.study2, args.pipeline, args.model_name, args.maps_name)
        for _, row1 in df1.iterrows():
            param = row1["parameter"]
            match = df2[df2["parameter"] == param]
            if match.empty:
                print(f"WARNING: '{param}' not found for {pid}/{args.study2}, skipping.")
                continue
            session1[param].append(float(row1["mean"]))
            session2[param].append(float(match["mean"].iloc[0]))
            patient_ids_per_param[param].append(pid)

    rows = []
    for param in sorted(session1):
        s1 = np.array(session1[param])
        s2 = np.array(session2[param])
        stats = compute_icc_pingouin(s1, s2)
        rows.append({
            "model": args.model_name,
            "parameter": param,
            "n_patients": len(s1),
            "icc": stats["icc"],
            "ci_lower": stats["ci_lower"],
            "ci_upper": stats["ci_upper"],
        })

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False, float_format="{:.6f}".format))

    save_dir = os.path.join(output_dir, "analysis", args.pipeline, args.maps_name, args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    result_df.to_csv(os.path.join(save_dir, "intra_repeat.csv"), index=False)
    print(f"\nSaved intra-patient repeatability: {save_dir}/intra_repeat.csv")

    parameters = sorted(session1)
    suptitle = f"Intra-patient {args.maps_name} ({args.model_name}) — {args.study1} vs {args.study2}"
    plot_paired_sessions(
        session1, session2, patient_ids_per_param, parameters,
        suptitle=suptitle,
        save_path=os.path.join(save_dir, "intra_paired.png"),
    )
    plot_pooled_sessions(
        session1, session2, patient_ids_per_param, parameters,
        suptitle=suptitle,
        save_path=os.path.join(save_dir, "intra_pooled.png"),
    )
    plot_spearman_matrix(
        session1, session2, patient_ids_per_param, parameters,
        suptitle=suptitle,
        save_path=os.path.join(save_dir, "intra_spearman.png"),
    )


if __name__ == "__main__":
    main()
