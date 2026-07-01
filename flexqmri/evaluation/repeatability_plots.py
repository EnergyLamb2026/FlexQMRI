"""Plotting utilities for intra-patient repeatability outlier inspection."""

import os
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


def _iqr_outlier_mask(values: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Return a boolean mask of values outside ``k * IQR`` from the quartiles.

    Args:
        values (np.ndarray): 1-D array of values.
        k (float, optional): IQR multiplier. Defaults to 1.5.

    Returns:
        np.ndarray: Boolean mask, ``True`` where the value is an outlier.
    """
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return (values < q1 - k * iqr) | (values > q3 + k * iqr)


def _save(fig, suptitle: str, save_path: str, dpi: int = 150, tight_rect=None) -> None:
    """Apply ``suptitle``, tighten layout, save the figure, and close it.

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        suptitle (str): Figure title.
        save_path (str): PNG output path; the parent directory is created if needed.
        dpi (int, optional): Resolution in dots per inch. Defaults to 150.
        tight_rect (sequence, optional): ``rect`` passed to ``tight_layout`` to reserve
            space for figure-level elements (e.g. ``[0, 0.15, 1, 0.93]``). Defaults to None.

    Returns:
        None
    """
    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout(rect=tight_rect)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def _plot_paired_axis(ax, s1: np.ndarray, s2: np.ndarray, patient_ids: Sequence[str], title: str) -> None:
    """Draw paired session-1/session-2 lines for one parameter onto ``ax``.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        s1 (np.ndarray): Mean ROI values from session 1, one per patient.
        s2 (np.ndarray): Mean ROI values from session 2, one per patient.
        patient_ids (Sequence[str]): Patient identifiers, same order as ``s1``/``s2``.
        title (str): Subplot title.

    Returns:
        None
    """
    for i, pid in enumerate(patient_ids):
        ax.plot([1, 2], [s1[i], s2[i]], marker="o", alpha=0.6)
        ax.annotate(pid, xy=(2.02, s2[i]), fontsize=7, va="center")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["S1", "S2"])
    ax.set_xlim(0.7, 2.4)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Mean ROI value")
    ax.grid(axis="y", alpha=0.3)


def _plot_pooled_axis(ax, values: np.ndarray, patient_ids: Sequence[str], label: str, title: str) -> None:
    """Draw a box + jittered strip plot for one session onto ``ax`` and label IQR outliers.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        values (np.ndarray): Mean ROI values, one per patient.
        patient_ids (Sequence[str]): Patient identifiers, same order as ``values``.
        label (str): X-tick label (e.g. ``"S1"``).
        title (str): Subplot title.

    Returns:
        None
    """
    ax.boxplot(values, widths=0.4, showfliers=False)
    rng = np.random.default_rng(0)
    x = 1 + rng.uniform(-0.06, 0.06, size=len(values))
    ax.scatter(x, values, alpha=0.7)
    mask = _iqr_outlier_mask(values)
    for k, pid in enumerate(patient_ids):
        if mask[k]:
            ax.annotate(pid, xy=(x[k] + 0.05, values[k]), fontsize=7, va="center", color="red")
    ax.set_xticks([1])
    ax.set_xticklabels([label])
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Mean ROI value")
    ax.grid(axis="y", alpha=0.3)


def plot_paired_sessions(
    session1: Mapping[str, Sequence[float]],
    session2: Mapping[str, Sequence[float]],
    patient_ids: Mapping[str, Sequence[str]],
    parameters: Sequence[str],
    suptitle: str,
    save_path: str,
) -> None:
    """Plot paired S1→S2 lines per patient, one subplot per parameter.

    Args:
        session1 (Mapping[str, Sequence[float]]): Maps parameter name to patient means in session 1.
        session2 (Mapping[str, Sequence[float]]): Maps parameter name to patient means in session 2.
        patient_ids (Mapping[str, Sequence[str]]): Maps parameter name to patient IDs aligned with
            the values in ``session1``/``session2``.
        parameters (Sequence[str]): Parameter names to plot, one column each.
        suptitle (str): Figure title.
        save_path (str): PNG output path; the parent directory is created if needed.

    Returns:
        None

    Notes:
        Each line connects one patient's S1 value to their S2 value. Patient IDs are
        annotated next to the S2 endpoint; deviant slopes flag repeatability outliers.
    """
    n = len(parameters)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), squeeze=False)
    for j, param in enumerate(parameters):
        s1 = np.asarray(session1[param], dtype=float)
        s2 = np.asarray(session2[param], dtype=float)
        _plot_paired_axis(axes[0, j], s1, s2, patient_ids[param], param)
    _save(fig, suptitle, save_path)


def plot_pooled_sessions(
    session1: Mapping[str, Sequence[float]],
    session2: Mapping[str, Sequence[float]],
    patient_ids: Mapping[str, Sequence[str]],
    parameters: Sequence[str],
    suptitle: str,
    save_path: str,
) -> None:
    """Plot pooled patient means with IQR outlier labels: one row per parameter, two columns (S1, S2).

    Args:
        session1 (Mapping[str, Sequence[float]]): Maps parameter name to patient means in session 1.
        session2 (Mapping[str, Sequence[float]]): Maps parameter name to patient means in session 2.
        patient_ids (Mapping[str, Sequence[str]]): Maps parameter name to patient IDs aligned with
            the values in ``session1``/``session2``.
        parameters (Sequence[str]): Parameter names; one subplot row each.
        suptitle (str): Figure title.
        save_path (str): PNG output path; the parent directory is created if needed.

    Returns:
        None

    Notes:
        Each subplot shows a boxplot with a jittered strip overlay. Patients whose
        value falls outside 1.5 × IQR are annotated in red.
    """
    n = len(parameters)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n), squeeze=False)
    for i, param in enumerate(parameters):
        s1 = np.asarray(session1[param], dtype=float)
        s2 = np.asarray(session2[param], dtype=float)
        _plot_pooled_axis(axes[i, 0], s1, patient_ids[param], "S1", f"{param} — S1")
        _plot_pooled_axis(axes[i, 1], s2, patient_ids[param], "S2", f"{param} — S2")
    _save(fig, suptitle, save_path)


def plot_spearman_matrix(
    session1: Mapping[str, Sequence[float]],
    session2: Mapping[str, Sequence[float]],
    patient_ids: Mapping[str, Sequence[str]],
    parameters: Sequence[str],
    suptitle: str,
    save_path: str,
) -> None:
    """Plot a Spearman correlation heatmap between parameters using per-patient session means.

    Args:
        session1 (Mapping[str, Sequence[float]]): Maps parameter name to patient means in session 1.
        session2 (Mapping[str, Sequence[float]]): Maps parameter name to patient means in session 2.
        patient_ids (Mapping[str, Sequence[str]]): Maps parameter name to patient IDs aligned with
            the values in ``session1``/``session2``.
        parameters (Sequence[str]): Parameter names; one row and column each.
        suptitle (str): Figure title.
        save_path (str): PNG output path; the parent directory is created if needed.

    Returns:
        None

    Notes:
        Each patient's value is the mean of their two session measurements. Only patients
        present in all parameters are included. The matrix is computed with
        ``scipy.stats.spearmanr`` on all pairs independently.
    """
    patient_data: dict[str, dict[str, float]] = {}
    for param in parameters:
        pids = patient_ids[param]
        s1 = np.asarray(session1[param], dtype=float)
        s2 = np.asarray(session2[param], dtype=float)
        for pid, v1, v2 in zip(pids, s1, s2):
            patient_data.setdefault(pid, {})[param] = (v1 + v2) / 2.0

    common = [pid for pid, d in patient_data.items() if all(p in d for p in parameters)]
    n_params = len(parameters)
    mat = np.array([[patient_data[pid][p] for p in parameters] for pid in common])

    rho = np.full((n_params, n_params), np.nan)
    for i in range(n_params):
        for j in range(n_params):
            r, _ = spearmanr(mat[:, i], mat[:, j])
            rho[i, j] = r

    fig, ax = plt.subplots(figsize=(max(4, n_params * 1.4), max(4, n_params * 1.4)))
    im = ax.imshow(rho, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    ax.set_xticks(range(n_params))
    ax.set_yticks(range(n_params))
    ax.set_xticklabels(parameters, rotation=45, ha="right")
    ax.set_yticklabels(parameters)
    for i in range(n_params):
        for j in range(n_params):
            if not np.isnan(rho[i, j]):
                ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(rho[i, j]) > 0.6 else "black")
    _save(fig, suptitle, save_path)


def _resolve_model_order(model_order: list | dict) -> list:
    """Resolve the global (union) model order from a flat list or per-param dict.

    Args:
        model_order (list | dict): Either a flat list of model keys shared across
            all parameters, or a dict mapping each param key to its own ordered list.

    Returns:
        list: Order-preserving union of model keys.
    """
    if not isinstance(model_order, dict):
        return list(model_order)
    global_model_order: list = []
    for keys in model_order.values():
        for k in keys:
            if k not in global_model_order:
                global_model_order.append(k)
    return global_model_order


def _models_for_param(model_order: list | dict, param: str, global_model_order: list) -> list:
    """Return the ordered model keys to draw for one parameter.

    Args:
        model_order (list | dict): Flat list or per-param dict of model keys.
        param (str): Parameter key.
        global_model_order (list): Fallback order when ``model_order`` is a flat list.

    Returns:
        list: Model keys for this parameter.
    """
    return model_order.get(param, global_model_order) if isinstance(model_order, dict) else model_order


def _draw_value_box_axis(ax, param: str, param_models: list, data: dict, color_map: dict, param_labels: dict, yticks=None) -> None:
    """Draw pooled S1+S2 value boxplots for one parameter onto ``ax``.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        param (str): Parameter key.
        param_models (list): Model keys defining box order.
        data (dict): ``{param: {model: {"S1": [...], "S2": [...]}}}`` per-patient ROI means.
        color_map (dict): Maps model key to face colour.
        param_labels (dict): Display label for each parameter key.
        yticks (Sequence[float], optional): Explicit y-tick positions in raw data units.
            When given, the axis limits are set to its first and last value. When None,
            3 round ticks are computed automatically. Defaults to None.

    Returns:
        None

    Notes:
        With automatic ticks the y-axis shows exactly 3 round, evenly spaced ticks.
        Only small magnitudes (tick exponent <= -2) are factored out into a ×10^n term
        appended to the subplot title; larger values keep their raw tick labels.
    """
    from matplotlib.ticker import MaxNLocator

    box_data = []
    for model_key in param_models:
        entry = data.get(param, {}).get(model_key, {})
        box_data.append(entry.get("S1", []) + entry.get("S2", []))

    bp = ax.boxplot(
        box_data,
        positions=range(len(param_models)),
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, markerfacecolor="none", markeredgewidth=0.8, alpha=0.7),
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, model_key in zip(bp["boxes"], param_models):
        patch.set_facecolor(color_map[model_key])
        patch.set_alpha(0.8)

    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, bottom=True)

    if yticks is not None:
        ticks = np.asarray(yticks, dtype=float)
    else:
        lo, hi = ax.get_ylim()
        locator = MaxNLocator(nbins=2)
        ticks = np.asarray(locator.tick_values(lo, hi))
        if len(ticks) < 3:
            ticks = np.linspace(lo, hi, 3)
        else:
            step = (ticks[1] - ticks[0]) * int(np.ceil((len(ticks) - 1) / 2))
            ticks = ticks[0] + step * np.arange(3)
    ax.set_ylim(ticks[0], ticks[-1])
    ax.set_yticks(ticks)
    ax.tick_params(axis="y", labelsize=8)

    title = param_labels.get(param, param)
    max_abs = float(np.max(np.abs(ticks))) if len(ticks) > 0 and np.any(ticks != 0) else 0.0
    if max_abs > 0:
        exp = int(np.floor(np.log10(max_abs)))
        if exp <= -2:
            scale = 10.0 ** exp
            ax.set_yticklabels([f"{t / scale:.2g}" for t in ticks])
            title = f"{title} ($\\times 10^{{-{abs(exp)}}}$)"
    ax.set_title(title, fontsize=9)


def _draw_icc_axis(ax, param: str, param_models: list, icc_data: dict, color_map: dict, style: str) -> None:
    """Draw per-model ICC with 95% CI for one parameter onto ``ax``.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        param (str): Parameter key.
        param_models (list): Model keys defining bar/point order.
        icc_data (dict): ``{param: {model: {"icc", "ci_lower", "ci_upper"}}}``.
        color_map (dict): Maps model key to colour.
        style (str): ``"bar"`` for coloured bars or ``"point"`` for dot-whisker markers.

    Returns:
        None

    Notes:
        Models without an ICC entry (e.g. fewer than 3 patients) are skipped, leaving
        a gap at their position so colours stay aligned with the value subplots.
        The y-axis is fixed to ``[0, 1]`` with ticks at 0, 0.5, and 1.
    """
    positions, iccs, lower_err, upper_err, colors = [], [], [], [], []
    for pos, model_key in enumerate(param_models):
        entry = icc_data.get(param, {}).get(model_key)
        if entry is None:
            continue
        positions.append(pos)
        iccs.append(entry["icc"])
        lower_err.append(max(0.0, entry["icc"] - entry["ci_lower"]))
        upper_err.append(max(0.0, entry["ci_upper"] - entry["icc"]))
        colors.append(color_map[model_key])

    yerr = np.array([lower_err, upper_err]) if positions else None
    if style == "bar":
        ax.bar(positions, iccs, width=0.6, color=colors, alpha=0.8,
               edgecolor="black", linewidth=0.8,
               yerr=yerr, capsize=2, error_kw=dict(linewidth=0.8))
    else:
        for pos, icc, le, ue, color in zip(positions, iccs, lower_err, upper_err, colors):
            ax.errorbar(pos, icc, yerr=[[le], [ue]], fmt="o", color=color,
                        markersize=4, capsize=2, elinewidth=0.8)

    ax.set_xlim(-0.6, len(param_models) - 0.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, bottom=True)


def _model_legend(fig, global_model_order: list, color_map: dict, model_labels: dict, y: float = 0.0) -> None:
    """Add a figure-level model colour legend along the bottom.

    Args:
        fig (matplotlib.figure.Figure): Figure to attach the legend to.
        global_model_order (list): Model keys in display order.
        color_map (dict): Maps model key to colour.
        model_labels (dict): Display label for each model key.
        y (float, optional): Vertical anchor of the legend in figure coordinates;
            larger values sit closer to the axes. Defaults to 0.0.

    Returns:
        None
    """
    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(facecolor=color_map[m], alpha=0.8, label=model_labels.get(m, m))
        for m in global_model_order
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(global_model_order),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, y),
    )


def plot_model_boxplots(
    data: dict,
    param_order: list,
    param_labels: dict,
    model_order: list | dict,
    model_labels: dict,
    suptitle: str,
    save_path: str,
    param_yticks: dict | None = None,
) -> None:
    """Plot model comparison boxplots across parameters, one subplot per parameter.

    Args:
        data (dict): ``{param: {model: {"S1": [...], "S2": [...]}}}`` — per-patient ROI means.
        param_order (list): Parameter keys defining subplot order (left to right).
        param_labels (dict): Display label (with units) for each parameter key.
        model_order (list | dict): Model keys defining box order within each subplot.
            Either a flat list (shared across all params) or a dict mapping each param key
            to its own ordered list of model keys (e.g. to show vendor only for R2*).
        model_labels (dict): Display label for each model key.
        suptitle (str): Figure suptitle.
        save_path (str): Output PNG path; parent directory is created if needed.
        param_yticks (dict, optional): Maps a parameter key to explicit y-tick positions
            (raw data units) for its subplot. Parameters absent from the dict use
            automatic 3-tick scaling. Defaults to None.

    Returns:
        None

    Notes:
        S1 and S2 values are pooled into a single box per model.  Model identity is
        conveyed by colour via a figure-level legend; the x-axis carries no labels.
        Each y-axis shows exactly 3 tick marks.  When the magnitude requires it, tick
        labels are rescaled and the power of ten is appended to the subplot title.
        Figure is sized for LaTeX \\textwidth (7.2 × 3.2 inches) and saved at 300 dpi.
        When ``model_order`` is a dict, the legend shows the union of all per-param
        model keys (order-preserving); colours are consistent across subplots.
    """
    sns.set_style("ticks")
    param_yticks = param_yticks or {}
    global_model_order = _resolve_model_order(model_order)
    palette = sns.color_palette("Set2", len(global_model_order))
    color_map = {m: palette[i] for i, m in enumerate(global_model_order)}

    n_params = len(param_order)
    fig, axes = plt.subplots(1, n_params, figsize=(7.2, 3.2), squeeze=False)
    for j, param in enumerate(param_order):
        param_models = _models_for_param(model_order, param, global_model_order)
        _draw_value_box_axis(axes[0, j], param, param_models, data, color_map, param_labels,
                             yticks=param_yticks.get(param))

    _model_legend(fig, global_model_order, color_map, model_labels)
    _save(fig, suptitle, save_path, dpi=300, tight_rect=[0, 0.12, 1, 0.93])


def plot_model_comparison_with_icc(
    data: dict,
    icc_data: dict,
    param_order: list,
    param_labels: dict,
    model_order: list | dict,
    model_labels: dict,
    suptitle: str,
    save_path: str,
    icc_style: str = "bar",
    param_yticks: dict | None = None,
) -> None:
    """Plot value boxplots (top row) and per-model ICC (bottom row).

    Args:
        data (dict): ``{param: {model: {"S1": [...], "S2": [...]}}}`` — per-patient ROI means.
        icc_data (dict): ``{param: {model: {"icc", "ci_lower", "ci_upper"}}}`` repeatability.
        param_order (list): Parameter keys defining column order within each block.
        param_labels (dict): Display label (with units) for each parameter key.
        model_order (list | dict): Model keys defining bar/box order; flat list shared
            across params or a per-param dict (e.g. to show vendor only for R2*).
        model_labels (dict): Display label for each model key.
        suptitle (str): Figure suptitle.
        save_path (str): Output PNG path; parent directory is created if needed.
        icc_style (str, optional): ``"bar"`` for coloured bars or ``"point"`` for
            dot-whisker markers. Defaults to ``"bar"``.
        param_yticks (dict, optional): Maps a parameter key to explicit y-tick positions
            (raw data units) for its value subplot. Parameters absent from the dict use
            automatic 3-tick scaling. Defaults to None.

    Returns:
        None

    Notes:
        The figure is a 2 × ``n_params`` grid: the top row holds value boxplots and the
        bottom row holds per-model ICC, with parameters aligned column-wise. Parameter
        titles label the top row only. Model identity is conveyed by colour via a shared
        figure-level legend; the leftmost axis of each row carries a row label
        ("Value" / "ICC"). Sized for LaTeX \\textwidth (7.2 × 4.0 inches) and saved at
        300 dpi.
    """
    sns.set_style("ticks")
    param_yticks = param_yticks or {}
    global_model_order = _resolve_model_order(model_order)
    palette = sns.color_palette("Set2", len(global_model_order))
    color_map = {m: palette[i] for i, m in enumerate(global_model_order)}

    n = len(param_order)
    fig, axes = plt.subplots(2, n, figsize=(7.2, 4.0), squeeze=False)

    for j, param in enumerate(param_order):
        param_models = _models_for_param(model_order, param, global_model_order)
        _draw_value_box_axis(axes[0, j], param, param_models, data, color_map, param_labels,
                             yticks=param_yticks.get(param))
        _draw_icc_axis(axes[1, j], param, param_models, icc_data, color_map, icc_style)

    axes[0, 0].set_ylabel("Value", fontsize=9)
    axes[1, 0].set_ylabel("ICC", fontsize=9)

    _model_legend(fig, global_model_order, color_map, model_labels, y=0.04)
    _save(fig, suptitle, save_path, dpi=300, tight_rect=[0, 0.10, 1, 0.93])
