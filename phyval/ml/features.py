"""Feature engineering shared by the ML modules, plus a small plotting helper."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # render to files, no display needed
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

MODULATIONS = ["BPSK", "QPSK", "QAM16", "QAM64"]
CHANNELS = ["AWGN", "PED_A", "VEH_A"]
NOISE_LABEL = "noise_limited"            # failures on bug-free builds (marginal tests / random noise)

# Consistent colours across all plots
PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#64748b"]


def triage_label(fault: pd.Series) -> pd.Series:
    return fault.replace({"none": NOISE_LABEL})


def triage_features(df: pd.DataFrame, limits: dict) -> pd.DataFrame:
    """Numeric features describing *symptoms* of a failing test (no build/fault information)."""
    X = pd.DataFrame(index=df.index)
    lim = df["modulation"].map(limits["evm_pct"])
    X["evm_over_limit"] = df["evm_pct"] / lim
    X["evm_db"] = df["evm_db"]
    X["log_ber"] = np.log10(df["ber"] + 1e-5)
    X["log_sat_ratio"] = np.log10(df["sat_ratio"] + 1e-5)
    X["abs_cpe_slope"] = df["cpe_slope_rad"].abs()
    X["evm_slope_db"] = df["evm_slope_db"]
    X["sc_spread"] = df["sc_spread"]
    X["image_corr"] = df["image_corr"]
    X["group_delay"] = df["group_delay"]
    X["chan_min_db"] = df["chan_min_db"]
    X["pilot_evm_db"] = df["pilot_evm_db"]
    X["abs_cfo_est"] = df["cfo_est"].abs()
    X["snr_db"] = df["snr_db"]
    X["cfo"] = df["cfo"]
    for m in MODULATIONS:
        X[f"mod_{m}"] = (df["modulation"] == m).astype(int)
    for c in CHANNELS:
        X[f"ch_{c}"] = (df["channel"] == c).astype(int)
    return X


def save_fig(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def style_axes(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def plot_confusion(cm: np.ndarray, labels: list, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else "#111827")
    ax.set_xlabel("Predicted root cause")
    ax.set_ylabel("True root cause")
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    return save_fig(fig, path)
