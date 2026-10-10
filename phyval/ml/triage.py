"""
Automatic failure triage.

Given a failing test, predict its root cause. Three models are compared:

  * metric_lr   - Logistic Regression on the numeric diagnostics (simple baseline)
  * metric_rf   - Random Forest on the numeric diagnostics
  * log_nlp     - NLP model: TF-IDF (Term Frequency - Inverse Document Frequency)
                  features of the normalised log text + Logistic Regression
  * combined    - soft vote: average of metric_rf and log_nlp probabilities

The model with the best cross-validated macro-F1 is the one deployed for triage.

Evaluation uses GroupKFold grouped by build: every failure from one build is in
the same fold, so the model is always tested on builds it has never seen. A
random row split would leak (failures within a build share one root cause and
look alike), and would report an inflated accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..framework.logs import normalize_log, parse_build_log
from .features import PALETTE, plot_confusion, save_fig, style_axes, triage_features, triage_label

MODEL_NAMES = {"metric_lr": "Logistic Regression (metrics)", "metric_rf": "Random Forest (metrics)",
               "log_nlp": "TF-IDF + LogReg (log text)", "combined": "Combined (RF + NLP)"}


def load_failure_texts(log_dir: Path, build_ids) -> dict:
    """Parse build logs from disk and return {(build_id, test_id): normalised text} for failures."""
    texts = {}
    for b in sorted(set(build_ids)):
        path = Path(log_dir) / f"{b}.log"
        if not path.exists():
            continue
        for rec in parse_build_log(path):
            if rec.get("status") == "FAIL":
                texts[(rec["build_id"], rec["test_id"])] = normalize_log(rec["text"])
    return texts


def build_triage_dataset(runs: pd.DataFrame, log_dir: Path) -> pd.DataFrame:
    fails = runs[runs["status"] == "FAIL"].copy()
    texts = load_failure_texts(log_dir, fails["build_id"].unique())
    fails["log_text"] = [texts.get((b, t), "") for b, t in zip(fails["build_id"], fails["test_id"])]
    fails["label"] = triage_label(fails["fault"])
    return fails.reset_index(drop=True)


def _metric_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


def _metric_rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                  n_jobs=-1, random_state=0)


def _log_nlp():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=3000, C=5.0, class_weight="balanced"))


def _aligned_proba(model, X, classes: list) -> np.ndarray:
    """predict_proba with columns in the global class order (a fold may lack a class)."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(classes)))
    for j, c in enumerate(model.classes_):
        out[:, classes.index(c)] = p[:, j]
    return out


def cross_validate(ds: pd.DataFrame, limits: dict, n_splits: int = 5) -> dict:
    X = triage_features(ds, limits)
    text = ds["log_text"].values
    y = ds["label"].values
    groups = ds["build_id"].values
    classes = sorted(np.unique(y).tolist())
    n_splits = min(n_splits, len(np.unique(groups)))
    oof = {k: np.zeros((len(ds), len(classes))) for k in ("metric_lr", "metric_rf", "log_nlp")}
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        for name, make, feats in (("metric_lr", _metric_lr, X), ("metric_rf", _metric_rf, X), ("log_nlp", _log_nlp, text)):
            m = make()
            Xtr = feats.iloc[tr] if hasattr(feats, "iloc") else feats[tr]
            Xte = feats.iloc[te] if hasattr(feats, "iloc") else feats[te]
            m.fit(Xtr, y[tr])
            oof[name][te] = _aligned_proba(m, Xte, classes)
    oof["combined"] = (oof["metric_rf"] + oof["log_nlp"]) / 2

    results = {"classes": classes, "n_failures": int(len(ds)), "n_builds": int(len(np.unique(groups))),
               "cv": f"GroupKFold by build, {n_splits} folds", "models": {}}
    preds = {}
    for name, P in oof.items():
        pred = np.array(classes)[P.argmax(1)]
        preds[name] = pred
        results["models"][name] = {"accuracy": float(accuracy_score(y, pred)),
                                   "macro_f1": float(f1_score(y, pred, average="macro"))}

    # Model selection: the model with the best cross-validated macro-F1 is deployed.
    selected = max(results["models"], key=lambda n: results["models"][n]["macro_f1"])
    results["selected_model"] = selected

    # Build-level triage: sum the per-failure probabilities of each build and take the top class.
    P = oof[selected]
    build_rows = []
    for b, idx in ds.groupby("build_id").indices.items():
        score = P[idx].sum(0)
        build_rows.append({"build_id": b, "true": y[idx][0], "pred": classes[int(score.argmax())],
                           "confidence": float(score.max() / score.sum()), "n_failures": len(idx)})
    bdf = pd.DataFrame(build_rows)
    results["build_level"] = {"accuracy": float((bdf["true"] == bdf["pred"]).mean()), "n_builds": int(len(bdf)),
                              "accuracy_buggy_builds": float((bdf[bdf["true"] != "noise_limited"]["true"] ==
                                                              bdf[bdf["true"] != "noise_limited"]["pred"]).mean())}
    results["per_class_recall"] = {c: float((preds[selected][y == c] == c).mean()) for c in classes}
    results["confusion_selected"] = confusion_matrix(y, preds[selected], labels=classes).tolist()
    return results, bdf


def fit_final(ds: pd.DataFrame, limits: dict, selected: str = "metric_rf") -> dict:
    """Train every model on all available failures; `selected` is the one used for predictions."""
    X = triage_features(ds, limits)
    y = ds["label"].values
    return {"metric_lr": _metric_lr().fit(X, y), "metric_rf": _metric_rf().fit(X, y),
            "log_nlp": _log_nlp().fit(ds["log_text"].values, y),
            "selected": selected, "classes": sorted(np.unique(y).tolist()),
            "feature_columns": list(X.columns), "limits": limits}


def _bundle_proba(bundle: dict, fails: pd.DataFrame) -> np.ndarray:
    classes = bundle["classes"]
    X = triage_features(fails, bundle["limits"])[bundle["feature_columns"]]
    text = fails["log_text"].values
    p = {"metric_lr": lambda: _aligned_proba(bundle["metric_lr"], X, classes),
         "metric_rf": lambda: _aligned_proba(bundle["metric_rf"], X, classes),
         "log_nlp": lambda: _aligned_proba(bundle["log_nlp"], text, classes)}
    if bundle["selected"] == "combined":
        return (p["metric_rf"]() + p["log_nlp"]()) / 2
    return p[bundle["selected"]]()


def predict(bundle: dict, fails: pd.DataFrame) -> pd.DataFrame:
    """Predict root cause for each failing test. `fails` needs metric columns and `log_text`."""
    classes = bundle["classes"]
    P = _bundle_proba(bundle, fails)
    out = fails[["build_id", "test_id", "modulation", "snr_db", "channel", "cfo", "evm_pct", "ber"]].copy()
    out["predicted_cause"] = np.array(classes)[P.argmax(1)]
    out["confidence"] = P.max(1)
    for j, c in enumerate(classes):
        out[f"p_{c}"] = P[:, j]
    return out


def summarize_build(pred: pd.DataFrame) -> dict:
    """Aggregate per-test predictions into one root-cause call for the build."""
    pcols = [c for c in pred.columns if c.startswith("p_")]
    score = pred[pcols].sum(0)
    total = float(score.sum()) or 1.0
    ranked = sorted(((c[2:], float(v / total)) for c, v in score.items()), key=lambda kv: -kv[1])
    return {"root_cause": ranked[0][0], "confidence": ranked[0][1], "ranking": ranked[:3]}


def run(ds: pd.DataFrame, limits: dict, paths, log=print) -> dict:
    """`ds` comes from build_triage_dataset(): failing tests with metrics, log text and labels."""
    log(f"  triage dataset: {len(ds)} failing tests from {ds['build_id'].nunique()} builds")
    results, bdf = cross_validate(ds, limits)
    for name, r in results["models"].items():
        log(f"  {MODEL_NAMES[name]:32s} accuracy={r['accuracy']:.3f}  macro-F1={r['macro_f1']:.3f}")
    log(f"  selected model: {MODEL_NAMES[results['selected_model']]}")
    log(f"  build-level root-cause accuracy: {results['build_level']['accuracy']:.3f}")

    bundle = fit_final(ds, limits, results["selected_model"])
    joblib.dump(bundle, paths.models / "triage.joblib")

    # ---- plots ----
    classes = results["classes"]
    results["plots"] = {}
    results["plots"]["confusion"] = str(plot_confusion(np.array(results["confusion_selected"]), classes,
                                                       f"Triage confusion: {MODEL_NAMES[results['selected_model']]}",
                                                       paths.plots / "triage_confusion.png"))
    imp = pd.Series(bundle["metric_rf"].feature_importances_, index=bundle["feature_columns"]).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(imp.index, imp.values, color=PALETTE[0])
    style_axes(ax, "Which diagnostics drive the triage decision", "Random Forest importance")
    results["plots"]["importance"] = str(save_fig(fig, paths.plots / "triage_importance.png"))

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = list(results["models"])
    f1s = [results["models"][n]["macro_f1"] for n in names]
    accs = [results["models"][n]["accuracy"] for n in names]
    xs = np.arange(len(names))
    ax.bar(xs - 0.18, accs, 0.36, label="Accuracy", color=PALETTE[0])
    ax.bar(xs + 0.18, f1s, 0.36, label="Macro-F1", color=PALETTE[3])
    ax.set_xticks(xs, [MODEL_NAMES[n].replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax, "Triage model comparison (cross-validated on unseen builds)")
    results["plots"]["models"] = str(save_fig(fig, paths.plots / "triage_models.png"))

    bdf.to_csv(paths.data / "triage_build_level.csv", index=False)
    (paths.data / "triage_results.json").write_text(json.dumps(results, indent=2))
    return results
