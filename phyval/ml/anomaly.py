"""
Build-level anomaly detection (predictive, early warning).

Pass/fail limits only catch a bug once it is bad enough to break a limit. A mild
bug can degrade the modem while every test still passes. This module learns what
"normal" looks like from the known-good baseline builds, then scores each later
build on how unusual its overall telemetry is.

Each build is summarised by one vector: pass rates, median EVM per modulation and
the medians of the receiver diagnostics. An Isolation Forest is trained on the
baseline builds only. A build scoring above every baseline build is flagged.

Reported:
  * ROC-AUC  (Receiver Operating Characteristic - Area Under Curve) of the anomaly
    score for separating buggy from clean builds (1.0 perfect, 0.5 random)
  * precision / recall at the threshold
  * how many "quiet" bugs were caught - buggy builds whose pass rate stayed >= 95%,
    i.e. bugs the pass/fail limits alone would very likely miss
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .features import CHANNELS, MODULATIONS, PALETTE, save_fig, style_axes

QUIET_PASS_RATE = 0.95


def build_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (bnum, bid), g in runs.groupby(["build_num", "build_id"]):
        r = {"build_num": bnum, "build_id": bid, "fault": g["fault"].iloc[0], "severity": g["severity"].iloc[0],
             "pass_rate": float((g["status"] == "PASS").mean())}
        for c in CHANNELS:
            gc = g[g["channel"] == c]
            r[f"pass_rate_{c}"] = float((gc["status"] == "PASS").mean()) if len(gc) else 1.0
        for m in MODULATIONS:
            gm = g[g["modulation"] == m]
            r[f"median_evm_db_{m}"] = float(gm["evm_db"].median()) if len(gm) else 0.0
        r["mean_log_sat"] = float(np.log10(g["sat_ratio"].mean() + 1e-6))
        r["median_abs_cpe_slope"] = float(g["cpe_slope_rad"].abs().median())
        r["median_evm_slope"] = float(g["evm_slope_db"].median())
        r["median_image_corr"] = float(g["image_corr"].median())
        r["median_group_delay"] = float(g["group_delay"].median())
        r["median_sc_spread"] = float(g["sc_spread"].median())
        r["median_pilot_evm_db"] = float(g["pilot_evm_db"].median())
        rows.append(r)
    return pd.DataFrame(rows).sort_values("build_num").reset_index(drop=True)


def _feature_cols(summary: pd.DataFrame) -> list:
    return [c for c in summary.columns if c not in ("build_num", "build_id", "fault", "severity")]


def run(runs: pd.DataFrame, baseline_builds: int, paths, log=print) -> dict:
    s = build_summary(runs)
    cols = _feature_cols(s)
    base = s["build_num"] < baseline_builds
    scaler = StandardScaler().fit(s.loc[base, cols])
    forest = IsolationForest(n_estimators=400, random_state=0).fit(scaler.transform(s.loc[base, cols]))
    s["anomaly_score"] = -forest.decision_function(scaler.transform(s[cols]))      # higher = more unusual
    threshold = float(s.loc[base, "anomaly_score"].max())
    s["flagged"] = s["anomaly_score"] > threshold
    s["buggy"] = s["fault"] != "none"

    ev = s[~base]
    y, flag = ev["buggy"].values, ev["flagged"].values
    auc = float(roc_auc_score(y, ev["anomaly_score"])) if len(np.unique(y)) > 1 else float("nan")
    tp = int((flag & y).sum())
    quiet = ev[ev["buggy"] & (ev["pass_rate"] >= QUIET_PASS_RATE)]
    res = {"roc_auc": auc, "threshold": threshold,
           "precision": float(tp / max(flag.sum(), 1)), "recall": float(tp / max(y.sum(), 1)),
           "false_alarms": int((flag & ~y).sum()), "clean_builds_evaluated": int((~y).sum()),
           "buggy_builds_evaluated": int(y.sum()),
           "quiet_bugs": int(len(quiet)), "quiet_bugs_flagged": int(quiet["flagged"].sum()),
           "quiet_bug_list": quiet[["build_id", "fault", "severity", "pass_rate", "flagged"]].to_dict("records")}
    log(f"  anomaly ROC-AUC={auc:.3f} precision={res['precision']:.2f} recall={res['recall']:.2f}; "
        f"quiet bugs (pass rate >= {int(QUIET_PASS_RATE*100)}%) flagged: {res['quiet_bugs_flagged']}/{res['quiet_bugs']}")

    joblib.dump({"scaler": scaler, "forest": forest, "columns": cols, "threshold": threshold},
                paths.models / "anomaly.joblib")

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    clean = s[~s["buggy"]]
    bug = s[s["buggy"]]
    axes[0].plot(s["build_num"], 100 * s["pass_rate"], color="#9ca3af", lw=1)
    axes[0].scatter(clean["build_num"], 100 * clean["pass_rate"], s=14, color=PALETTE[2], label="clean build")
    axes[0].scatter(bug["build_num"], 100 * bug["pass_rate"], s=18, color=PALETTE[1], label="buggy build")
    style_axes(axes[0], "Regression pass rate per build", "", "pass rate (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[1].plot(s["build_num"], s["anomaly_score"], color="#9ca3af", lw=1)
    axes[1].scatter(clean["build_num"], clean["anomaly_score"], s=14, color=PALETTE[2])
    axes[1].scatter(bug["build_num"], bug["anomaly_score"], s=18, color=PALETTE[1])
    q = s[s["buggy"] & (s["pass_rate"] >= QUIET_PASS_RATE)]
    axes[1].scatter(q["build_num"], q["anomaly_score"], s=80, facecolors="none", edgecolors=PALETTE[4],
                    label=f"quiet bug (pass rate >= {int(QUIET_PASS_RATE*100)}%)")
    axes[1].axhline(threshold, color=PALETTE[0], ls="--", lw=1, label="alarm threshold")
    axes[1].axvspan(-0.5, baseline_builds - 0.5, color=PALETTE[2], alpha=0.07, label="baseline (training) builds")
    style_axes(axes[1], "Isolation Forest anomaly score", "build number", "anomaly score")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    res["plot"] = str(save_fig(fig, paths.plots / "anomaly_timeline.png"))

    s.to_csv(paths.data / "build_summary.csv", index=False)
    (paths.data / "anomaly_results.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def score_build(bundle: dict, runs: pd.DataFrame) -> dict:
    s = build_summary(runs)
    for c in bundle["columns"]:
        if c not in s:
            s[c] = 0.0
    score = float(-bundle["forest"].decision_function(bundle["scaler"].transform(s[bundle["columns"]]))[0])
    return {"anomaly_score": score, "threshold": bundle["threshold"], "flagged": score > bundle["threshold"]}
