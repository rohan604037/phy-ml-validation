"""
PHYVAL command-line interface.

    python -m phyval all              # whole pipeline: qualify -> regress -> analyze -> report
    python -m phyval all --quick      # smaller regression history (about 1 minute)
    python -m phyval run-build --fault cfo_uncorrected --severity 0.6 --budget 0.3
    python -m phyval triage-log --log output/logs/B0042.log
    python -m phyval faults           # list injectable faults

Run `python -m phyval <command> --help` for every option.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import DEFAULT_OUTPUT, Paths, load_config


def log(msg: str):
    print(msg, flush=True)


def _need(path: Path, hint: str):
    if not path.exists():
        sys.exit(f"Missing {path}.\nRun `{hint}` first.")


# -----------------------------------------------------------------------------
# Stages
# -----------------------------------------------------------------------------
def stage_qualify(cfg, paths):
    from .framework.runner import generate_candidates, qualify
    t = time.time()
    log("[1/4] Qualifying candidate tests against the golden design")
    cands = generate_candidates(cfg)
    qual = qualify(cfg, cands, progress=log)
    qual.to_csv(paths.data / "test_plan_qualified.csv", index=False)
    log(f"  {int(qual['qualified'].sum())} of {len(qual)} candidate tests qualified ({time.time() - t:.1f}s)")
    return qual


def stage_regress(cfg, paths, quick=False):
    from .framework.runner import make_build_schedule, run_regression
    _need(paths.data / "test_plan_qualified.csv", "python -m phyval qualify")
    qual = pd.read_csv(paths.data / "test_plan_qualified.csv")
    tests = qual[qual["qualified"]]
    n = cfg["regression"]["quick_n_builds"] if quick else cfg["regression"]["n_builds"]
    t = time.time()
    log(f"[2/4] Running regression history: {n} builds x {len(tests)} tests")
    for old in paths.logs.glob("B*.log"):
        old.unlink()
    sched = make_build_schedule(cfg, n)
    sched.to_csv(paths.data / "builds.csv", index=False)
    runs = run_regression(cfg, sched, tests, paths.logs, progress=log)
    runs.to_csv(paths.data / "runs.csv", index=False)
    log(f"  {len(runs):,} test executions, {int((runs['status'] == 'FAIL').sum()):,} failures ({time.time() - t:.1f}s)")
    return runs


def stage_analyze(cfg, paths):
    from .ml import anomaly, discovery, selection, triage
    _need(paths.data / "runs.csv", "python -m phyval regress")
    runs = pd.read_csv(paths.data / "runs.csv")
    t = time.time()
    log("[3/4] Machine learning")
    log(" - auto-triage (supervised classification, metrics + NLP on logs)")
    ds = triage.build_triage_dataset(runs, paths.logs)
    tri = triage.run(ds, cfg["limits"], paths, log)
    log(" - discovery of new failure types (unsupervised clustering, open-set detection)")
    disc = discovery.run(ds, cfg["limits"], paths, log)
    log(" - intelligent test selection (gradient boosting)")
    sel = selection.run(runs, cfg["modules"], paths, log)
    log(" - build anomaly detection (Isolation Forest)")
    ano = anomaly.run(runs, cfg["regression"]["baseline_builds"], paths, log)
    log(f"  done ({time.time() - t:.1f}s)")
    return tri, disc, sel, ano


def _load_json(paths, name):
    _need(paths.data / name, "python -m phyval analyze")
    d = json.loads((paths.data / name).read_text())
    # re-point plot paths at this output folder (the folder may have been moved or copied)
    for key in ("plot",):
        if key in d:
            d[key] = str(paths.plots / Path(d[key]).name)
    if "plots" in d:
        d["plots"] = {k: str(paths.plots / Path(v).name) for k, v in d["plots"].items()}
    return d


def stage_report(cfg, paths, email_to=None):
    from .report import html as rep
    from .report.notify import publish
    from .report.phy_plots import constellation_grid, golden_evm_curve
    log("[4/4] Building report")
    qual = pd.read_csv(paths.data / "test_plan_qualified.csv")
    builds = pd.read_csv(paths.data / "builds.csv")
    runs = pd.read_csv(paths.data / "runs.csv")
    tri, disc = _load_json(paths, "triage_results.json"), _load_json(paths, "discovery_results.json")
    sel, ano = _load_json(paths, "selection_results.json"), _load_json(paths, "anomaly_results.json")
    phy = {"constellations": constellation_grid(cfg, paths), "golden_evm": golden_evm_curve(qual, paths)}
    out = rep.regression_report(cfg, qual, builds, runs, tri, disc, sel, ano, phy, paths.reports / "regression_report.html")
    log(f"  report: {out}")
    best = tri["models"][tri["selected_model"]]
    summary = (f"<h2>PHY validation regression summary</h2><ul>"
               f"<li>{len(builds)} builds, {len(runs):,} test executions</li>"
               f"<li>Auto-triage macro-F1 on unseen builds: {100 * best['macro_f1']:.1f}%</li>"
               f"<li>Test selection: {100 * sel['ml']['bug_detection_at_budget']:.0f}% of buggy builds caught with "
               f"{int(100 * sel['budget_point'])}% of test time</li>"
               f"<li>Anomaly detection ROC-AUC: {ano['roc_auc']:.2f}</li></ul><p>Full report attached.</p>")
    publish(f"[PHYVAL] Regression report {datetime.now():%Y-%m-%d}", summary, out, paths.reports, email_to, log)
    return out


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------
def cmd_all(args, cfg, paths):
    t = time.time()
    stage_qualify(cfg, paths)
    stage_regress(cfg, paths, args.quick)
    stage_analyze(cfg, paths)
    out = stage_report(cfg, paths, args.email_to)
    log(f"\nAll done in {time.time() - t:.0f}s. Open the report:\n  {out}")


def cmd_run_build(args, cfg, paths):
    """Validate one new build: pick tests, run them, triage failures, report."""
    from .framework.logs import normalize_log, parse_build_log
    from .framework.runner import run_build
    from .ml import anomaly, selection, triage
    from .phy.ofdm import FAULTS
    from .report import html as rep
    from .report.notify import publish

    for f in ("test_plan_qualified.csv",):
        _need(paths.data / f, "python -m phyval all --quick")
    for f in ("triage.joblib", "selection.joblib", "anomaly.joblib"):
        _need(paths.models / f, "python -m phyval all --quick")
    if args.fault == "random":
        # Nightly mode: a reproducible "surprise" build. The seed (e.g. the date) decides
        # whether there is a bug, which one, and how severe.
        rng = np.random.default_rng(args.seed)
        names = list(FAULTS)
        args.fault = names[int(rng.integers(len(names)))]
        args.severity = 0.0 if args.fault == "none" else float(np.round(rng.uniform(0.1, 1.0), 2))
        if not args.changed:
            extra = rng.choice(cfg["modules"], int(rng.integers(0, 3)), replace=False).tolist()
            args.changed = ",".join(sorted(set(extra + [cfg["fault_module"].get(args.fault, "scheduler")])))
        log(f"Random build: injected fault hidden from the models (revealed in the report).")
    if args.fault not in FAULTS:
        sys.exit(f"Unknown fault '{args.fault}'. Choose from: {', '.join(FAULTS)} or random")

    qual = pd.read_csv(paths.data / "test_plan_qualified.csv")
    suite = qual[qual["qualified"]]
    changed = [m.strip() for m in args.changed.split(",")] if args.changed else []
    if not changed:
        changed = [cfg["fault_module"].get(args.fault, "scheduler")]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    build = {"build_num": 50_000 + int(np.random.default_rng().integers(0, 1_000_000)) if args.seed is None else 50_000 + args.seed,
             "build_id": args.build_id or f"NEW_{stamp}", "date": datetime.now().date().isoformat(),
             "fault": args.fault, "severity": args.severity, "changed_modules": ",".join(sorted(changed))}

    if args.budget < 1.0:
        sel_bundle = joblib.load(paths.models / "selection.joblib")
        ranked = selection.rank_for_build(sel_bundle, suite, changed)
        idx = selection.select_tests(ranked["p_fail"].values, ranked["cost_s"].values, args.budget * suite["cost_s"].sum())
        tests = ranked.iloc[idx].sort_values("test_num")
        log(f"ML test selection: {len(tests)} of {len(suite)} tests chosen for a {int(args.budget * 100)}% time budget "
            f"(changed modules: {', '.join(changed)})")
    else:
        tests = suite
        log(f"Running full suite: {len(tests)} tests")

    log_dir = paths.root / "new_builds"
    results = run_build(cfg, build, tests, log_dir)
    n_fail = int((results["status"] == "FAIL").sum())
    log(f"{build['build_id']}: {len(results) - n_fail} passed, {n_fail} failed")

    pred, summary = None, None
    if n_fail:
        texts = {r["test_id"]: normalize_log(r["text"]) for r in parse_build_log(log_dir / f"{build['build_id']}.log")
                 if r.get("status") == "FAIL"}
        fails = results[results["status"] == "FAIL"].copy()
        fails["log_text"] = fails["test_id"].map(texts).fillna("")
        bundle = joblib.load(paths.models / "triage.joblib")
        pred = triage.predict(bundle, fails)
        summary = triage.summarize_build(pred)
        log(f"Predicted root cause: {summary['root_cause']} (confidence {100 * summary['confidence']:.0f}%)")
        for cause, share in summary["ranking"]:
            log(f"   {cause:18s} {100 * share:5.1f}%")
        if summary["confidence"] < 0.6:
            log("   Low confidence: possibly a new kind of failure - route to an engineer.")
    else:
        log("No failures - nothing to triage.")

    ano = None
    if args.budget >= 1.0:
        ano = anomaly.score_build(joblib.load(paths.models / "anomaly.joblib"), results)
        log(f"Anomaly check: {'ANOMALOUS' if ano['flagged'] else 'normal'} "
            f"(score {ano['anomaly_score']:.3f}, threshold {ano['threshold']:.3f})")
    else:
        log("Anomaly check skipped (it needs the full suite, run with --budget 1.0).")

    out = rep.build_report(build, results, len(suite), pred, summary, ano, paths.reports / f"build_{build['build_id']}.html")
    results.to_csv(log_dir / f"{build['build_id']}_results.csv", index=False)
    if pred is not None:
        pred.to_csv(log_dir / f"{build['build_id']}_triage.csv", index=False)
    log(f"Build report: {out}")
    head = (f"<h2>Build {build['build_id']}</h2><p>{len(results)} tests, {n_fail} failures. "
            + (f"Predicted root cause: <b>{summary['root_cause']}</b> ({100 * summary['confidence']:.0f}%)." if summary else "All passed.")
            + "</p>")
    publish(f"[PHYVAL] Build {build['build_id']}: {n_fail} failures", head, out, paths.reports, args.email_to, log)


def cmd_triage_log(args, cfg, paths):
    """Triage a raw build log using only the NLP (text) model."""
    from .framework.logs import normalize_log, parse_build_log
    _need(paths.models / "triage.joblib", "python -m phyval all --quick")
    bundle = joblib.load(paths.models / "triage.joblib")
    recs = [r for r in parse_build_log(args.log) if r.get("status") == "FAIL"]
    if not recs:
        log("No failing tests in this log.")
        return
    nlp = bundle["log_nlp"]
    texts = [normalize_log(r["text"]) for r in recs]
    P = nlp.predict_proba(texts)
    classes = list(nlp.classes_)
    for r, p in zip(recs, P):
        log(f"{r['test_id']}  {r['modulation']:5s} {r['channel']:5s} snr={r['snr_db']:4.0f} cfo={r['cfo']:.2f}  ->  "
            f"{classes[int(p.argmax())]:16s} ({100 * p.max():.0f}%)")
    tot = P.sum(0) / P.sum()
    order = np.argsort(-tot)
    log(f"\nBuild-level root cause from {len(recs)} failures (log text only):")
    for j in order[:3]:
        log(f"   {classes[j]:18s} {100 * tot[j]:5.1f}%")


def cmd_index(args, cfg, paths):
    """Write reports/index.html: a landing page linking every report (used for GitHub Pages)."""
    import html as _h
    from .report.html import page
    reports = sorted(paths.reports.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    reports = [r for r in reports if r.name != "index.html"]
    rows = []
    for r in reports:
        kind = "Regression report" if r.name.startswith("regression") else "Build report"
        when = datetime.fromtimestamp(r.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        rows.append(f"<tr><td><a href='{_h.escape(r.name)}'>{_h.escape(r.stem)}</a></td><td>{kind}</td><td>{when}</td></tr>")
    body = ("<p>Generated automatically by the PHYVAL CI/CD pipeline.</p>"
            "<div class='wrap'><table><thead><tr><th>Report</th><th>Type</th><th>Generated (UTC)</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")
    out = paths.reports / "index.html"
    out.write_text(page("PHYVAL Validation Reports", f"Last updated {datetime.now():%Y-%m-%d %H:%M} UTC", body),
                   encoding="utf-8")
    log(f"Index: {out} ({len(reports)} reports)")


def cmd_faults(args, cfg, paths):
    from .phy.ofdm import FAULTS
    for k, v in FAULTS.items():
        log(f"  {k:16s} [{cfg['fault_module'].get(k, '-'):11s}] {v}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m phyval", description="ML-driven validation of an OFDM baseband PHY")
    p.add_argument("--config", default=None, help="path to the YAML test plan (default: config/test_plan.yaml)")
    p.add_argument("--out", default=str(DEFAULT_OUTPUT), help="output folder (default: ./output)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("all", help="run the whole pipeline")
    a.add_argument("--quick", action="store_true", help="shorter regression history (faster)")
    a.add_argument("--email-to", default=None, help="email the report (needs PHYVAL_SMTP_* variables)")
    sub.add_parser("qualify", help="stage 1: qualify tests on the golden design")
    r = sub.add_parser("regress", help="stage 2: simulate the build history")
    r.add_argument("--quick", action="store_true")
    sub.add_parser("analyze", help="stage 3: train and evaluate all ML models")
    rp = sub.add_parser("report", help="stage 4: build the HTML report")
    rp.add_argument("--email-to", default=None)

    b = sub.add_parser("run-build", help="validate one new build end to end")
    b.add_argument("--fault", default="none", help="bug to inject (see `faults`), or `random`; default none")
    b.add_argument("--severity", type=float, default=0.6, help="bug severity 0..1 (default 0.6)")
    b.add_argument("--changed", default="", help="comma-separated changed modules (default: the bug's module)")
    b.add_argument("--budget", type=float, default=1.0, help="fraction of full-suite test time to spend (default 1.0)")
    b.add_argument("--build-id", default=None)
    b.add_argument("--seed", type=int, default=None, help="fix the noise seed for a repeatable run")
    b.add_argument("--email-to", default=None)

    t = sub.add_parser("triage-log", help="triage a raw build log file with the NLP model")
    t.add_argument("--log", required=True)
    sub.add_parser("faults", help="list the injectable faults")
    sub.add_parser("index", help="write an index.html linking all reports")

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    cfg["_path"] = args.config or "config/test_plan.yaml"
    paths = Paths(Path(args.out))
    {"all": cmd_all, "qualify": lambda a, c, p_: stage_qualify(c, p_),
     "regress": lambda a, c, p_: stage_regress(c, p_, a.quick), "analyze": lambda a, c, p_: stage_analyze(c, p_),
     "report": lambda a, c, p_: stage_report(c, p_, a.email_to), "run-build": cmd_run_build,
     "triage-log": cmd_triage_log, "faults": cmd_faults, "index": cmd_index}[args.cmd](args, cfg, paths)


if __name__ == "__main__":
    main()
