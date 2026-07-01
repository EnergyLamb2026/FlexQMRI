"""Repeatability and variability metrics for MRI parameter maps."""

import os

import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats


def load_stats(
    output_dir: str,
    patient_id: str,
    study: str,
    pipeline: str,
    model_name: str,
    maps_name: str,
) -> pd.DataFrame:
    """Load a per-patient stats CSV produced by save_roi_stats.

    Args:
        output_dir (str): Root output directory.
        patient_id (str): Patient identifier.
        study (str): Study/session folder name.
        pipeline (str): Pipeline label (e.g. 'standard').
        model_name (str): Model identifier used as the innermost folder level.
        maps_name (str): Map type identifier (e.g. 'ivim', 'r2star').

    Returns:
        pd.DataFrame: Stats table for this patient and session.

    Raises:
        FileNotFoundError: If the stats CSV does not exist.
    """
    path = os.path.join(output_dir, patient_id, study, pipeline, maps_name, model_name, "stats.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stats CSV not found: {path}")
    return pd.read_csv(path)


def compute_icc(
    session1: np.ndarray,
    session2: np.ndarray,
    confidence: float = 0.95,
) -> dict:
    """Compute ICC(2,1): two-way random effects, absolute agreement, single measures.

    Args:
        session1 (np.ndarray): Mean ROI values from session 1, shape (n_subjects,).
        session2 (np.ndarray): Mean ROI values from session 2, shape (n_subjects,).
        confidence (float, optional): Confidence level for the CI. Defaults to 0.95.

    Returns:
        dict: Dictionary with keys ``icc``, ``ci_lower``, and ``ci_upper``.

    Raises:
        ValueError: If ``session1`` and ``session2`` have different lengths or fewer
            than 3 subjects.

    Notes:
        Uses the two-way random-effects ANOVA decomposition (Shrout & Fleiss, 1979).
        The point estimate is ICC(2,1) for absolute agreement.  The CI is derived
        from the F-distribution bounds on MS_B/MS_E, with the between-session
        correction term retained so the bounds stay consistent with the ICC(2,1)
        formula rather than collapsing to ICC(3,1).
    """
    if len(session1) != len(session2):
        raise ValueError("session1 and session2 must have the same length.")
    n = len(session1)
    if n < 3:
        raise ValueError("At least 3 subjects are required to compute ICC.")

    k = 2
    y = np.column_stack([session1, session2])  # (n, k)
    grand_mean = y.mean()
    subject_means = y.mean(axis=1)  # (n,)
    session_means = y.mean(axis=0)  # (k,)

    ss_b = k * np.sum((subject_means - grand_mean) ** 2)
    ss_w = np.sum((y - subject_means[:, None]) ** 2)
    ss_r = n * np.sum((session_means - grand_mean) ** 2)
    ss_e = ss_w - ss_r

    ms_b = ss_b / (n - 1)
    ms_r = ss_r / (k - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))

    icc = (ms_b - ms_e) / (ms_b + (k - 1) * ms_e + (k / n) * (ms_r - ms_e))

    # F-based CI bounds on ms_b / ms_e
    alpha = 1 - confidence
    df_b = n - 1
    df_e = (n - 1) * (k - 1)
    f1 = ms_b / ms_e
    f_l = f1 / scipy.stats.f.ppf(1 - alpha / 2, df_b, df_e)
    f_u = f1 * scipy.stats.f.ppf(1 - alpha / 2, df_e, df_b)

    # Session-effect correction term (keeps bounds on the ICC(2,1) scale)
    correction = (k / n) * (ms_r / ms_e - 1)
    ci_lower = float(np.clip((f_l - 1) / (f_l + (k - 1) + correction), 0.0, 1.0))
    ci_upper = float(np.clip((f_u - 1) / (f_u + (k - 1) + correction), 0.0, 1.0))

    return {"icc": float(icc), "ci_lower": ci_lower, "ci_upper": ci_upper}


def compute_icc_pingouin(session1: np.ndarray, session2: np.ndarray) -> dict:
    """Compute ICC(A,1) using pingouin's intraclass_corr.

    Args:
        session1 (np.ndarray): Mean ROI values from session 1, shape (n_subjects,).
        session2 (np.ndarray): Mean ROI values from session 2, shape (n_subjects,).

    Returns:
        dict: Dictionary with keys ``icc``, ``ci_lower``, and ``ci_upper``.
            CI bounds are at the 95% confidence level (pingouin default).

    Raises:
        ValueError: If ``session1`` and ``session2`` have different lengths or fewer
            than 3 subjects.

    Notes:
        Returns the ``ICC(A,1)`` row from ``pingouin.intraclass_corr`` (two-way
        random effects, absolute agreement, single rater). This is equivalent to
        Shrout & Fleiss's ICC(2,1) — the same definition as :func:`compute_icc`,
        but delegated to pingouin for robustness.
    """
    if len(session1) != len(session2):
        raise ValueError("session1 and session2 must have the same length.")
    n = len(session1)
    if n < 3:
        raise ValueError("At least 3 subjects are required to compute ICC.")

    long_df = pd.DataFrame({
        "subject": np.tile(np.arange(n), 2),
        "session": np.repeat([1, 2], n),
        "value": np.concatenate([session1, session2]),
    })
    icc_df = pg.intraclass_corr(
        data=long_df,
        targets="subject",
        raters="session",
        ratings="value",
        nan_policy="omit",
    ).set_index("Type")

    row = icc_df.loc["ICC(A,1)"]
    ci_lower, ci_upper = row["CI95"]
    return {
        "icc": float(row["ICC"]),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def compute_inter_patient_stats(values: np.ndarray, confidence: float = 0.95) -> dict:
    """Compute mean, coefficient of variation, and CI of the mean across patients.

    Args:
        values (np.ndarray): One mean ROI value per patient, shape (n_patients,).
        confidence (float, optional): Confidence level for the CI. Defaults to 0.95.

    Returns:
        dict: Dictionary with keys ``mean``, ``cv``, ``ci_lower``, and ``ci_upper``.

    Raises:
        ValueError: If fewer than 2 values are provided.

    Notes:
        CV is expressed as a percentage: std / |mean| * 100.
        The CI uses the normal approximation: mean ± z * std / sqrt(n).
    """
    if len(values) < 2:
        raise ValueError("At least 2 patients are required.")

    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    cv = std / abs(mean) * 100 if mean != 0 else float("nan")
    z = scipy.stats.norm.ppf((1 + confidence) / 2.0)
    ci_length = z * std / np.sqrt(n)

    return {
        "mean": mean,
        "cv": cv,
        "ci_lower": mean - ci_length,
        "ci_upper": mean + ci_length,
    }
