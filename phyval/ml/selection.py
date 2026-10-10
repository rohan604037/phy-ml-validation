"""
Intelligent test selection.

Running the whole suite on every build is slow (on real RF equipment, each test
takes tens of seconds to minutes). Given the list of modules a build changed, this
model predicts each test's probability of failing and runs the most promising tests
first, within a time budget.

Model: Histogram Gradient Boosting classifier.
Features: test parameters (modulation, SNR = Signal-to-Noise Ratio, channel, CFO),
the build's changed modules, and the test's historical failure rate (computed only
from *earlier* builds, so no future information leaks in).

Evaluation is chronological: train on the first 70% of builds, evaluate on the
later 30% - exactly how the tool would be used in practice.

Ranking score = P(fail) / test cost (expected failures caught per second), a greedy
solution to the time-budget knapsack problem.

Compared against:
  * random      - random order (averaged over 30 shuffles)
  * history     - rank by historical failure rate only (a common heuristic)
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .features import CHANNELS, MODULATIONS, PALETTE, save_fig, style_axes

BUDGETS = np.round(np.arange(0.05, 1.0001, 0.05), 2)


def add_history(runs: pd.DataFrame) -> pd.DataFrame:
    """Failure rate of each test over all *previous* builds (0 for the first build)."""
    df = runs.sort_values(["build_num", "test_num"]).copy()
    df["failed"] = (df["status"] == "FAIL").astype(int)
    df["hist_fail_rate"] = (df.groupby("test_id")["failed"]
                            .transform(lambda s: s.shift().expanding().mean()).fillna(0.0))
    return df


def selection_features(df: pd.DataFrame, modules: list) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["snr_db"] = df["snr_db"]
    X["cfo"] = df["cfo"]
    X["cost_s"] = df["cost_s"]
    X["hist_fail_rate"] = df["hist_fail_rate"]
    for m in MODULATIONS:
        X[f"mod_{m}"] = (df["modulation"] == m).astype(int)
    for c in CHANNELS:
        X[f"ch_{c}"] = (df["channel"] == c).astype(int)
    changed = df["changed_modules"].fillna("").str.split(",")
    for mod in modules:
        X[f"changed_{mod}"] = changed.apply(lambda lst, m=mod: int(m in lst))
    return X


def _make_model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


def select_tests(scores: np.ndarray, costs: np.ndarray, budget_s: float) -> np.ndarray:
    """Greedy: highest score-per-second first, until the time budget is used up."""
    order = np.argsort(-(scores / costs), kind="stable")
    keep = np.cumsum(costs[order]) <= budget_s + 1e-9
    return order[keep]


def _evaluate(test_df: pd.DataFrame, score_col: str, rng=None, n_random: int = 30) -> pd.DataFrame:
    rows = []
    builds = list(test_df.groupby("build_id"))
    for budget in BUDGETS:
        caught = total = det = det_total = 0.0
        reps = n_random if score_col == "random" else 1
        for r in range(reps):
            for _, g in builds:
                costs = g["cost_s"].values
                scores = rng.random(len(g)) if score_col == "random" else g[score_col].values
                idx = select_tests(scores, costs, budget * costs.sum())
                failed = g["failed"].values
                caught += failed[idx].sum()
                total += failed.sum()
                if g["fault"].iloc[0] != "none" and failed.sum() > 0:
                    det_total += 1
                    det += failed[idx].sum() > 0
        rows.append({"budget": budget, "failure_recall": caught / max(total, 1),
                     "bug_detection": det / max(det_total, 1)})
    return pd.DataFrame(rows)


def run(runs: pd.DataFrame, modules: list, paths, log=print, train_frac: float = 0.7) -> dict:
    df = add_history(runs)
    X = selection_features(df, modules)
    y = df["failed"].values
    split = int(df["build_num"].max() * train_frac)
    tr = (df["build_num"] <= split).values
    te = ~tr

    model = _make_model().fit(X[tr], y[tr])
    test_df = df[te].copy()
    test_df["p_fail"] = model.predict_proba(X[te])[:, 1]
    test_df["hist"] = test_df["hist_fail_rate"] + 1e-6
    auc = float(roc_auc_score(y[te], test_df["p_fail"])) if len(np.unique(y[te])) > 1 else float("nan")
    log(f"  test-failure prediction ROC-AUC on later builds: {auc:.3f}")

    rng = np.random.default_rng(0)
    curves = {"ml": _evaluate(test_df, "p_fail"), "history": _evaluate(test_df, "hist"),
              "random": _evaluate(test_df, "random", rng)}

    def budget_for_full_detection(c):
        hit = c[c["bug_detection"] >= 0.999]
        return float(hit["budget"].min()) if len(hit) else 1.0

    at = 0.2
    summary = {"train_builds": int(df[tr]["build_id"].nunique()),
               "test_builds": int(test_df["build_id"].nunique()), "roc_auc": auc, "budget_point": at}
    for name, c in curves.items():
        row = c[np.isclose(c["budget"], at)].iloc[0]
        summary[name] = {"failure_recall_at_budget": float(row["failure_recall"]),
                         "bug_detection_at_budget": float(row["bug_detection"]),
                         "budget_for_100pct_bug_detection": budget_for_full_detection(c)}
        log(f"  {name:8s} @ {int(at*100)}% time: catches {100*row['failure_recall']:.1f}% of failures, "
            f"detects {100*row['bug_detection']:.1f}% of buggy builds; "
            f"100% detection needs {100*summary[name]['budget_for_100pct_bug_detection']:.0f}% of time")

    # Final model on all builds, plus the latest history per test, for use on new builds.
    final = _make_model().fit(X, y)
    hist_latest = df.groupby("test_id")["failed"].mean().to_dict()
    joblib.dump({"model": final, "feature_columns": list(X.columns), "modules": modules,
                 "hist_fail_rate": hist_latest}, paths.models / "selection.joblib")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = {"ml": "ML selection (gradient boosting)", "history": "History heuristic", "random": "Random"}
    colors = {"ml": PALETTE[0], "history": PALETTE[3], "random": PALETTE[7]}
    for name, c in curves.items():
        axes[0].plot(100 * c["budget"], 100 * c["failure_recall"], marker="o", ms=3, color=colors[name], label=labels[name])
        axes[1].plot(100 * c["budget"], 100 * c["bug_detection"], marker="o", ms=3, color=colors[name], label=labels[name])
    style_axes(axes[0], "Failing tests caught", "Test time used (% of full suite)", "% of failures caught")
    style_axes(axes[1], "Buggy builds detected (>= 1 failure caught)", "Test time used (% of full suite)", "% of buggy builds")
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.axvline(100 * at, color="#9ca3af", ls="--", lw=1)
    summary["plot"] = str(save_fig(fig, paths.plots / "selection_curves.png"))

    pd.concat([c.assign(strategy=n) for n, c in curves.items()]).to_csv(paths.data / "selection_curves.csv", index=False)
    (paths.data / "selection_results.json").write_text(json.dumps(summary, indent=2))
    return summary


def rank_for_build(bundle: dict, tests: pd.DataFrame, changed_modules: list) -> pd.DataFrame:
    """Score every test for a new build with the given changed modules."""
    df = tests.copy()
    df["changed_modules"] = ",".join(changed_modules)
    df["hist_fail_rate"] = df["test_id"].map(bundle["hist_fail_rate"]).fillna(0.0)
    X = selection_features(df, bundle["modules"])[bundle["feature_columns"]]
    df["p_fail"] = bundle["model"].predict_proba(X)[:, 1]
    df["score_per_s"] = df["p_fail"] / df["cost_s"]
    return df.sort_values("score_per_s", ascending=False)
