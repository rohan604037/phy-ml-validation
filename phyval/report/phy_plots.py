"""Signal-processing sanity plots: constellations per fault and the golden EVM-vs-SNR curve."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..ml.features import PALETTE, save_fig, style_axes
from ..phy.ofdm import FAULTS, FaultConfig, channel_taps, constellation, simulate_link


def constellation_grid(cfg: dict, paths, mod: str = "QAM16", channel: str = "VEH_A",
                       snr_db: float = 35, cfo: float = 0.15, severity: float = 0.7):
    prof = cfg["channels"][channel]
    h = channel_taps(prof["delays"], prof["powers_db"], np.random.default_rng([cfg["seed"], 5, 1]))
    h_old = channel_taps(prof["delays"], prof["powers_db"], np.random.default_rng([cfg["seed"], 5, 2]))
    ideal = constellation(mod)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    for ax, name in zip(axes.flat, FAULTS):
        fault = FaultConfig.from_name(name, 0.0 if name == "none" else severity)
        r = simulate_link(mod, snr_db, h, cfo, fault, np.random.default_rng(1), n_symbols=40,
                          h_stale=h_old, keep_symbols=True)
        s = r.rx_symbols
        ax.scatter(s.real, s.imag, s=3, alpha=0.35, color=PALETTE[0] if name == "none" else PALETTE[1])
        ax.scatter(ideal.real, ideal.imag, s=14, marker="+", color="#111827")
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}\nEVM {r.evm_pct:.1f}%", fontsize=9)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.02, 0.5, f"{mod}, {channel} channel\nSNR {snr_db} dB, CFO {cfo}\n"
                       f"fault severity {severity}\n\n+ = ideal constellation point",
                       fontsize=9, va="center")
    fig.suptitle("Received constellation for each injected fault", x=0.01, ha="left", fontweight="bold")
    return save_fig(fig, paths.plots / "phy_constellations.png")


def golden_evm_curve(qual, paths):
    """Mean golden-design EVM vs SNR for each channel, with the per-modulation limits."""
    fig, ax = plt.subplots(figsize=(6.6, 4))
    sub = qual[qual["cfo"] == 0.0]
    for i, ch in enumerate(sorted(sub["channel"].unique())):
        g = sub[sub["channel"] == ch].groupby("snr_db")["golden_evm_mean"].mean()
        ax.plot(g.index, g.values, marker="o", ms=4, color=PALETTE[i], label=f"{ch} (golden)")
    for j, (mod, lim) in enumerate(qual.groupby("modulation")["evm_limit"].first().items()):
        ax.axhline(lim, ls=":", lw=1, color="#6b7280")
        ax.text(sub["snr_db"].max(), lim, f" {mod} limit {lim}%", fontsize=7, va="bottom", ha="right", color="#374151")
    ax.set_yscale("log")
    style_axes(ax, "Golden-design EVM vs SNR (averaged over modulations)", "SNR (dB)", "EVM (%)")
    ax.legend(frameon=False, fontsize=8)
    return save_fig(fig, paths.plots / "phy_golden_evm.png")
