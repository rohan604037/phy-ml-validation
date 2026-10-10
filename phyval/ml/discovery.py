"""
Discovering failure types nobody has labelled yet (unsupervised learning).

Two experiments:

1. Clustering - K-Means groups failing tests by their diagnostic fingerprint
   with no labels at all. We then measure how well the clusters match the true
   root causes (ARI = Adjusted Rand Index, NMI = Normalised Mutual Information;
   1.0 = perfect agreement, 0 = random).

2. Open-set ("new bug") detection - for each bug type in turn, train the triage
   classifier as if that bug had never been seen, then show it failures of that
   bug. A well-behaved classifier should be *unsure* (low top probability) and
   flag them as "unknown - needs an engineer" instead of confidently mislabelling.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .features import NOISE_LABEL, PALETTE, save_fig, style_axes, triage_features

SYMPTOM_FEATURES = ["evm_over_limit", "log_ber", "log_sat_ratio", "abs_cpe_slope", "evm_slope_db",
                    "sc_spread", "image_corr", "group_delay", "chan_min_db", "pilot_evm_db"]
UNKNOWN_THRESHOLD = 0.6     # top-class probability below this -> "unknown"


def run(ds: pd.DataFrame, limits: dict, paths, log=print) -> dict:
    X_all = triage_features(ds, limits)
    y = ds["label"].values
    classes = sorted(np.unique(y).tolist())

    # ---------------- 1. clustering ----------------
    Z = StandardScaler().fit_transform(X_all[SYMPTOM_FEATURES])
    km = KMeans(n_clusters=len(classes), n_init=20, random_state=0).fit(Z)
    clusters = km.labels_
    ct = pd.crosstab(clusters, y)
    purity = float(ct.max(axis=1).sum() / len(y))
    res = {"kmeans": {"k": len(classes), "ari": float(adjusted_rand_score(y, clusters)),
                      "nmi": float(normalized_mutual_info_score(y, clusters)), "purity": purity,
                      "cluster_majority": {int(c): str(ct.loc[c].idxmax()) for c in ct.index}}}
    log(f"  K-Means clusters vs true causes: ARI={res['kmeans']['ari']:.3f} NMI={res['kmeans']['nmi']:.3f} purity={purity:.3f}")

    pcs = PCA(n_components=2, random_state=0).fit_transform(Z)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    for i, c in enumerate(classes):
        m = y == c
        axes[0].scatter(pcs[m, 0], pcs[m, 1], s=8, alpha=0.6, color=PALETTE[i % len(PALETTE)], label=c)
    for k in range(len(classes)):
        m = clusters == k
        axes[1].scatter(pcs[m, 0], pcs[m, 1], s=8, alpha=0.6, color=PALETTE[k % len(PALETTE)], label=f"cluster {k}")
    style_axes(axes[0], "True root cause", "PC 1", "PC 2")
    style_axes(axes[1], "K-Means clusters (found without labels)", "PC 1")
    axes[0].legend(fontsize=7, frameon=False, markerscale=2)
    axes[1].legend(fontsize=7, frameon=False, markerscale=2)
    res["plot"] = str(save_fig(fig, paths.plots / "discovery_clusters.png"))

    # ---------------- 2. open-set detection ----------------
    feats = X_all
    rows = []
    bugs = [c for c in classes if c != NOISE_LABEL]
    for f in bugs:
        known_mask = y != f
        known_idx = np.where(known_mask)[0]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
        tr, te = next(gss.split(known_idx, y[known_idx], ds["build_id"].values[known_idx]))
        tr, te = known_idx[tr], known_idx[te]
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                    n_jobs=-1, random_state=0).fit(feats.iloc[tr], y[tr])
        p_unseen = rf.predict_proba(feats[~known_mask]).max(1)
        p_known = rf.predict_proba(feats.iloc[te]).max(1)
        rows.append({"held_out_bug": f,
                     "unseen_flagged_pct": float(100 * np.mean(p_unseen < UNKNOWN_THRESHOLD)),
                     "known_flagged_pct": float(100 * np.mean(p_known < UNKNOWN_THRESHOLD)),
                     "unseen_median_conf": float(np.median(p_unseen)),
                     "known_median_conf": float(np.median(p_known))})
    open_set = pd.DataFrame(rows)
    res["open_set"] = open_set.to_dict("records")
    res["open_set_threshold"] = UNKNOWN_THRESHOLD
    log(f"  open-set: mean unseen-bug failures flagged {open_set['unseen_flagged_pct'].mean():.1f}% "
        f"vs known {open_set['known_flagged_pct'].mean():.1f}% (threshold {UNKNOWN_THRESHOLD})")

    (paths.data / "discovery_results.json").write_text(json.dumps(res, indent=2))
    return res
