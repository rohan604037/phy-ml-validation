"""Self-contained HTML reports (images embedded, so one file can be emailed or uploaded anywhere)."""
from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..phy.ofdm import FAULTS

CSS = """
:root{--bg:#f8fafc;--card:#ffffff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--accent:#2563eb;
--good:#16a34a;--bad:#dc2626;--warn:#d97706}
@media (prefers-color-scheme: dark){:root{--bg:#0b1120;--card:#111827;--ink:#e5e7eb;--muted:#94a3b8;
--line:#1f2937;--accent:#60a5fa}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 10px;padding-top:6px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:18px 0 6px}.sub{color:var(--muted);margin:0 0 20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .v{font-size:24px;font-weight:700}.card .k{color:var(--muted);font-size:12.5px}
img{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fff;margin:6px 0}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13.5px;background:var(--card)}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left}th{background:rgba(37,99,235,.08)}
.wrap{overflow-x:auto}.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600}
.pass{background:rgba(22,163,74,.15);color:var(--good)}.fail{background:rgba(220,38,38,.15);color:var(--bad)}
.note{background:var(--card);border-left:3px solid var(--accent);padding:10px 14px;border-radius:6px;margin:10px 0}
code{background:rgba(100,116,139,.15);padding:1px 5px;border-radius:4px;font-size:13px}
dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px}dt{font-weight:600}dd{margin:0;color:var(--muted)}
"""

GLOSSARY = {
    "OFDM": "Orthogonal Frequency-Division Multiplexing - the multi-carrier modulation used by Wi-Fi, LTE and 5G NR",
    "PHY": "Physical Layer - the part of the modem that turns bits into radio waveforms and back",
    "EVM": "Error Vector Magnitude - RMS distance between received and ideal constellation points (lower is better)",
    "BER": "Bit Error Rate - fraction of bits received wrongly",
    "SNR": "Signal-to-Noise Ratio, in decibels (dB)",
    "CFO": "Carrier Frequency Offset - frequency mismatch between transmitter and receiver oscillators",
    "CP": "Cyclic Prefix - copy of the symbol tail placed before it to absorb multipath echoes",
    "ADC": "Analog-to-Digital Converter",
    "I/Q": "In-phase / Quadrature - the two components of a complex baseband signal",
    "QPSK / QAM": "Quadrature Phase-Shift Keying / Quadrature Amplitude Modulation",
    "TF-IDF": "Term Frequency - Inverse Document Frequency, a way to turn text into numeric features",
    "ROC-AUC": "Area under the Receiver Operating Characteristic curve; 1.0 = perfect ranking, 0.5 = random",
    "ARI / NMI": "Adjusted Rand Index / Normalised Mutual Information - agreement between clusters and true labels",
    "F1": "Harmonic mean of precision and recall; macro-F1 averages it over all classes equally",
    "CI/CD": "Continuous Integration / Continuous Delivery",
}


def img(path) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img alt="{html.escape(Path(path).stem)}" src="data:image/png;base64,{data}">'


def card(value, label) -> str:
    return f'<div class="card"><div class="v">{html.escape(str(value))}</div><div class="k">{html.escape(label)}</div></div>'


def table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if c in fmt:
                v = fmt[c](v)
            s = str(v)
            if s in ("PASS", "FAIL"):
                s = f'<span class="pill {s.lower()}">{s}</span>'
            else:
                s = html.escape(s)
            cells.append(f"<td>{s}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def page(title: str, subtitle: str, body: str) -> str:
    gl = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in GLOSSARY.items())
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body><main><h1>{html.escape(title)}</h1><p class='sub'>{html.escape(subtitle)}</p>{body}"
            f"<h2>Abbreviations</h2><dl>{gl}</dl></main></body></html>")


pct = lambda v: f"{100 * v:.1f}%"   # noqa: E731


def regression_report(cfg, qual, builds, runs, triage, discovery, selection, anomaly, phy_plots, path: Path) -> Path:
    q = qual[qual["qualified"]]
    n_fail = int((runs["status"] == "FAIL").sum())
    sel_ml, sel_rand = selection["ml"], selection["random"]
    tri_best = triage["models"][triage["selected_model"]]
    b = []

    b.append('<div class="cards">' + "".join([
        card(f"{len(q)} / {len(qual)}", "tests qualified on golden design"),
        card(len(builds), "builds in regression history"),
        card(f"{len(runs):,}", "test executions"),
        card(f"{n_fail:,}", "failures to triage"),
        card(pct(tri_best["macro_f1"]), "triage macro-F1 (unseen builds)"),
        card(pct(triage["build_level"]["accuracy"]), "build-level root-cause accuracy"),
        card(pct(sel_ml["bug_detection_at_budget"]), f"buggy builds caught with {int(selection['budget_point']*100)}% of test time"),
        card(f"{anomaly['roc_auc']:.2f}", "anomaly-detection ROC-AUC"),
    ]) + "</div>")

    b.append("<div class='note'>Everything in this report was generated by <code>python -m phyval all</code>. "
             "Faults are injected into a simulated OFDM modem; every model is evaluated only on builds it "
             "never saw during training.</div>")

    # ---- PHY ----
    b.append("<h2>1. The device under test: OFDM baseband PHY</h2>")
    b.append("<p>A 64-subcarrier OFDM link (52 used, 4 pilots, 16-sample cyclic prefix) with preamble-based CFO "
             "estimation, least-squares channel estimation, zero-forcing equalisation and pilot phase tracking. "
             "Six realistic hardware/firmware bugs can be injected with a severity from 0 to 1:</p>")
    fdf = pd.DataFrame([{"fault": k, "owner module": cfg["fault_module"].get(k, "-"), "what goes wrong": v}
                        for k, v in FAULTS.items() if k != "none"])
    b.append(table(fdf))
    b.append(img(phy_plots["constellations"]))
    b.append(img(phy_plots["golden_evm"]))

    # ---- framework ----
    b.append("<h2>2. Spec-driven test automation</h2>")
    b.append(f"<p>The YAML test plan sweeps {len(cfg['sweep']['modulation'])} modulations x "
             f"{len(cfg['sweep']['snr_db'])} SNR points x {len(cfg['sweep']['channel'])} channels x "
             f"{len(cfg['sweep']['cfo'])} CFO values = {len(qual)} candidate tests. Each candidate was run "
             f"{cfg['qualification']['golden_runs']} times on the golden design and kept only if it always passed "
             f"({len(q)} qualified). Pass/fail limits: EVM per modulation and BER &le; {cfg['limits']['ber_max']:.0e}.</p>")
    cov = q.groupby(["modulation", "channel"]).size().unstack(fill_value=0).reset_index()
    b.append("<h3>Qualified tests per modulation and channel</h3>" + table(cov))
    fr = runs.groupby("fault").agg(builds=("build_id", "nunique"), tests_run=("status", "size"),
                                   fail_rate=("status", lambda s: (s == "FAIL").mean())).reset_index()
    b.append("<h3>Failure rate by injected bug</h3>" + table(fr, {"fail_rate": pct}))

    # ---- triage ----
    b.append("<h2>3. ML auto-triage of failures</h2>")
    b.append(f"<p>{triage['n_failures']:,} failing tests from {triage['n_builds']} builds. Cross-validation: "
             f"{html.escape(triage['cv'])} (all failures of a build stay together, so the score reflects "
             f"performance on brand-new builds). Deployed model: <b>{html.escape(triage['selected_model'])}</b>.</p>")
    mdf = pd.DataFrame([{"model": k, "accuracy": v["accuracy"], "macro_F1": v["macro_f1"]} for k, v in triage["models"].items()])
    b.append(table(mdf, {"accuracy": pct, "macro_F1": pct}))
    b.append(img(triage["plots"]["models"]) + img(triage["plots"]["confusion"]) + img(triage["plots"]["importance"]))
    rec = pd.DataFrame([{"root cause": k, "recall": v} for k, v in triage["per_class_recall"].items()])
    b.append("<h3>Per-class recall</h3>" + table(rec, {"recall": pct}))
    b.append(f"<p>Build-level call (summing per-failure probabilities across a build): "
             f"{pct(triage['build_level']['accuracy'])} of {triage['build_level']['n_builds']} builds with failures; "
             f"{pct(triage['build_level']['accuracy_buggy_builds'])} of genuinely buggy builds.</p>")

    # ---- discovery ----
    km = discovery["kmeans"]
    b.append("<h2>4. Unsupervised discovery of new failure types</h2>")
    b.append(f"<p>K-Means with k={km['k']} on standardised symptom features, without labels: "
             f"ARI {km['ari']:.2f}, NMI {km['nmi']:.2f}, purity {pct(km['purity'])}.</p>")
    b.append(img(discovery["plot"]))
    osd = pd.DataFrame(discovery["open_set"])
    b.append(f"<h3>Open-set test: can the classifier say \"I don't know\"?</h3><p>Each bug type is hidden from "
             f"training in turn. A prediction with top probability below {discovery['open_set_threshold']} is flagged "
             f"as unknown.</p>")
    b.append(table(osd, {"unseen_flagged_pct": lambda v: f"{v:.1f}%", "known_flagged_pct": lambda v: f"{v:.1f}%",
                         "unseen_median_conf": lambda v: f"{v:.2f}", "known_median_conf": lambda v: f"{v:.2f}"}))

    # ---- selection ----
    b.append("<h2>5. Intelligent test selection</h2>")
    b.append(f"<p>Trained on the first {selection['train_builds']} builds, evaluated on the next "
             f"{selection['test_builds']} (chronological split). Predicting which tests fail: ROC-AUC "
             f"{selection['roc_auc']:.3f}.</p>")
    sdf = pd.DataFrame([{"strategy": n, "failures caught @ budget": selection[n]["failure_recall_at_budget"],
                         "buggy builds detected @ budget": selection[n]["bug_detection_at_budget"],
                         "time for 100% bug detection": selection[n]["budget_for_100pct_bug_detection"]}
                        for n in ("ml", "history", "random")])
    b.append(f"<p>Budget = {int(selection['budget_point']*100)}% of full-suite execution time.</p>")
    b.append(table(sdf, {"failures caught @ budget": pct, "buggy builds detected @ budget": pct,
                         "time for 100% bug detection": pct}))
    b.append(img(selection["plot"]))
    b.append(f"<p>Random selection with the same budget catches {pct(sel_rand['failure_recall_at_budget'])} of failures.</p>")

    # ---- anomaly ----
    b.append("<h2>6. Predictive anomaly detection across builds</h2>")
    b.append(f"<p>Isolation Forest trained on {cfg['regression']['baseline_builds']} known-good builds. On later builds: "
             f"ROC-AUC {anomaly['roc_auc']:.3f}, precision {pct(anomaly['precision'])}, recall {pct(anomaly['recall'])}, "
             f"{anomaly['false_alarms']} false alarms on {anomaly['clean_builds_evaluated']} clean builds. "
             f"Quiet bugs (pass rate &ge; 95%, easy to miss by pass/fail alone): "
             f"{anomaly['quiet_bugs_flagged']} of {anomaly['quiet_bugs']} flagged.</p>")
    b.append(img(anomaly["plot"]))
    if anomaly["quiet_bug_list"]:
        b.append(table(pd.DataFrame(anomaly["quiet_bug_list"]), {"pass_rate": pct}))

    # ---- build list ----
    b.append("<h2>7. Build history</h2>")
    bl = runs.groupby(["build_id", "fault", "severity", "changed_modules"]).agg(
        pass_rate=("status", lambda s: (s == "PASS").mean()), failures=("status", lambda s: (s == "FAIL").sum())).reset_index()
    b.append(table(bl, {"pass_rate": pct}))

    sub = f"Generated {datetime.now():%Y-%m-%d %H:%M} | config: {Path(cfg.get('_path', 'config/test_plan.yaml')).name}"
    path.write_text(page("PHY Validation: ML Regression Report", sub, "".join(b)), encoding="utf-8")
    return path


def build_report(build: dict, results: pd.DataFrame, n_suite: int, triage_pred, summary, anomaly, path: Path) -> Path:
    n_fail = int((results["status"] == "FAIL").sum())
    b = ['<div class="cards">' + "".join([
        card(build["build_id"], "build"),
        card(f"{len(results)} / {n_suite}", "tests run / suite size"),
        card(f"{results['cost_s'].sum() / 60:.1f} min", "estimated lab time"),
        card(n_fail, "failures"),
        card(summary["root_cause"] if summary else "-", "predicted root cause"),
        card(pct(summary["confidence"]) if summary else "-", "triage confidence"),
    ]) + "</div>"]
    b.append(f"<p>Changed modules: <code>{html.escape(build['changed_modules'])}</code>. "
             f"Injected for this demo: <code>{html.escape(build['fault'])}</code> "
             f"(severity {build['severity']}). The triage models do not see this line.</p>")
    if summary:
        rank = pd.DataFrame(summary["ranking"], columns=["root cause", "share of evidence"])
        b.append("<h2>Root-cause ranking</h2>" + table(rank, {"share of evidence": pct}))
        if summary["confidence"] < 0.6:
            b.append("<div class='note'>Low confidence: this may be a failure type the models have not seen. "
                     "Route to an engineer.</div>")
    if anomaly:
        state = "ANOMALOUS" if anomaly["flagged"] else "normal"
        b.append(f"<h2>Anomaly check</h2><p>Build telemetry is <b>{state}</b> "
                 f"(score {anomaly['anomaly_score']:.3f}, threshold {anomaly['threshold']:.3f}).</p>")
    if triage_pred is not None and len(triage_pred):
        b.append("<h2>Failing tests</h2>")
        show = triage_pred[["test_id", "modulation", "snr_db", "channel", "cfo", "evm_pct", "ber", "predicted_cause", "confidence"]]
        b.append(table(show, {"evm_pct": lambda v: f"{v:.2f}", "ber": lambda v: f"{v:.1e}", "confidence": pct}))
    b.append("<h2>All executed tests</h2>")
    b.append(table(results[["test_id", "modulation", "snr_db", "channel", "cfo", "evm_pct", "ber", "status"]],
                   {"evm_pct": lambda v: f"{v:.2f}", "ber": lambda v: f"{v:.1e}"}))
    path.write_text(page(f"Build {build['build_id']}: validation summary",
                         f"Generated {datetime.now():%Y-%m-%d %H:%M}", "".join(b)), encoding="utf-8")
    return path
