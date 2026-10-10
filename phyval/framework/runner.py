"""
Test automation framework.

  1. generate_candidates()  - expand the YAML sweep into concrete test cases
  2. qualify()              - keep only tests the golden design passes reliably
  3. run_build()            - execute a test list against one build, write its log
  4. make_build_schedule()  - create a realistic history of builds (some buggy)
  5. run_regression()       - run every build in the schedule = regression history
"""
from __future__ import annotations

import itertools
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..phy.ofdm import FAULTS, FaultConfig, channel_taps, simulate_link
from .logs import render_test_log

RESULT_COLUMNS = ["evm_pct", "evm_db", "ber", "cfo_est", "sat_ratio", "evm_slope_db", "cpe_slope_rad",
                  "sc_spread", "image_corr", "group_delay", "pilot_evm_db", "chan_min_db"]


# -----------------------------------------------------------------------------
# Test plan
# -----------------------------------------------------------------------------
def test_cost(cfg: dict, mod: str, channel: str, cfo: float) -> float:
    c = cfg["cost_model"]
    cost = c["base_s"] * c["channel_factor"][channel] * c["modulation_factor"][mod]
    return round(cost + (c["cfo_extra_s"] if cfo else 0.0), 1)


def generate_candidates(cfg: dict) -> pd.DataFrame:
    sw = cfg["sweep"]
    rows = []
    for i, (mod, snr, ch, cfo) in enumerate(itertools.product(sw["modulation"], sw["snr_db"], sw["channel"], sw["cfo"])):
        rows.append({"test_num": i, "test_id": f"TC_{i:04d}", "modulation": mod, "snr_db": float(snr),
                     "channel": ch, "cfo": float(cfo), "cost_s": test_cost(cfg, mod, ch, cfo)})
    return pd.DataFrame(rows)


def _channels_for(cfg: dict, test: dict):
    """Each test case uses a fixed channel realisation (like a recorded emulator profile)."""
    prof = cfg["channels"][test["channel"]]
    seed = cfg["seed"]
    h = channel_taps(prof["delays"], prof["powers_db"], np.random.default_rng([seed, int(test["test_num"]), 1]))
    h_old = channel_taps(prof["delays"], prof["powers_db"], np.random.default_rng([seed, int(test["test_num"]), 2]))
    return h, h_old


def check_limits(cfg: dict, mod: str, evm_pct: float, ber: float) -> list:
    reasons = []
    lim = cfg["limits"]["evm_pct"][mod]
    if evm_pct > lim:
        reasons.append(f"EVM {evm_pct:.2f}% > limit {lim:.2f}% ({mod})")
    if ber > cfg["limits"]["ber_max"]:
        reasons.append(f"BER {ber:.2e} > limit {cfg['limits']['ber_max']:.0e}")
    return reasons


def execute_test(cfg: dict, test: dict, fault: FaultConfig, noise_seed) -> dict:
    """Run one test case and return its measurements + pass/fail."""
    h, h_old = _channels_for(cfg, test)
    res = simulate_link(test["modulation"], test["snr_db"], h, test["cfo"], fault,
                        np.random.default_rng(noise_seed), n_symbols=cfg["phy"]["n_data_symbols"], h_stale=h_old)
    out = {"evm_pct": res.evm_pct, "evm_db": res.evm_db, "ber": res.ber, **res.diagnostics}
    reasons = check_limits(cfg, test["modulation"], res.evm_pct, res.ber)
    out["status"] = "FAIL" if reasons else "PASS"
    out["reasons"] = reasons
    return out


def qualify(cfg: dict, candidates: pd.DataFrame, progress=None) -> pd.DataFrame:
    """Run each candidate on the golden design several times; keep those that always pass."""
    golden = FaultConfig.from_name("none")
    n_runs = cfg["qualification"]["golden_runs"]
    rows = []
    for idx, test in enumerate(candidates.to_dict("records")):
        evms, fails = [], 0
        for k in range(n_runs):
            r = execute_test(cfg, test, golden, [cfg["seed"], 900_000 + k, test["test_num"], 3])
            evms.append(r["evm_pct"])
            fails += r["status"] == "FAIL"
        limit = cfg["limits"]["evm_pct"][test["modulation"]]
        rows.append({**test, "golden_evm_mean": float(np.mean(evms)), "golden_evm_max": float(np.max(evms)),
                     "evm_limit": limit, "golden_margin_pct": float(limit - np.max(evms)),
                     "golden_fails": fails, "qualified": fails == 0})
        if progress and (idx + 1) % 50 == 0:
            progress(f"  qualified {idx + 1}/{len(candidates)} candidates")
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Builds
# -----------------------------------------------------------------------------
def make_build_schedule(cfg: dict, n_builds: int) -> pd.DataFrame:
    """History of builds. Baseline builds are clean; later builds carry a bug with some probability.

    Faults are assigned round-robin (then shuffled) so each fault type appears a similar
    number of times - otherwise a short history may contain no example of some bug.
    """
    reg = cfg["regression"]
    rng = np.random.default_rng([cfg["seed"], 7])
    modules = cfg["modules"]
    fault_names = [f for f in FAULTS if f != "none"]
    n_base = min(reg["baseline_builds"], n_builds)
    n_rest = n_builds - n_base
    n_faulty = int(round(n_rest * reg["fault_probability"]))
    faults = [fault_names[i % len(fault_names)] for i in range(n_faulty)] + ["none"] * (n_rest - n_faulty)
    faults = list(rng.permutation(faults))
    lo, hi = reg["severity_range"]
    rows = []
    t0 = datetime(2026, 6, 1, 2, 0, 0)
    for b in range(n_builds):
        fault = "none" if b < n_base else faults[b - n_base]
        severity = 0.0 if fault == "none" else float(np.round(rng.uniform(lo, hi), 3))
        n_changed = int(rng.integers(1, 4))
        changed = set(rng.choice(modules, n_changed, replace=False).tolist())
        if fault != "none":
            changed.add(cfg["fault_module"][fault])
        rows.append({"build_num": b, "build_id": f"B{b:04d}", "date": (t0 + timedelta(days=b)).date().isoformat(),
                     "baseline": b < n_base, "fault": fault, "severity": severity,
                     "changed_modules": ",".join(sorted(changed))})
    return pd.DataFrame(rows)


def run_build(cfg: dict, build: dict, tests: pd.DataFrame, log_dir: Path) -> pd.DataFrame:
    """Execute `tests` against one build and write logs/<build_id>.log."""
    fault = FaultConfig.from_name(build["fault"], build["severity"])
    rng_log = np.random.default_rng([cfg["seed"], int(build["build_num"]), 11])
    t = datetime.fromisoformat(build["date"] + "T02:00:00")
    rows, texts = [], []
    for test in tests.to_dict("records"):
        r = execute_test(cfg, test, fault, [cfg["seed"], int(build["build_num"]), int(test["test_num"]), 3])
        texts.append(render_test_log(test, build["build_id"], r, r["status"], r["reasons"], rng_log, t))
        t += timedelta(seconds=float(test["cost_s"]))
        rows.append({"build_num": build["build_num"], "build_id": build["build_id"],
                     "test_num": test["test_num"], "test_id": test["test_id"],
                     "modulation": test["modulation"], "snr_db": test["snr_db"], "channel": test["channel"],
                     "cfo": test["cfo"], "cost_s": test["cost_s"],
                     **{k: r[k] for k in RESULT_COLUMNS}, "status": r["status"],
                     "fault": build["fault"], "severity": build["severity"],
                     "changed_modules": build["changed_modules"]})
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{build['build_id']}.log").write_text("\n".join(texts) + "\n", encoding="utf-8")
    return pd.DataFrame(rows)


def run_regression(cfg: dict, schedule: pd.DataFrame, tests: pd.DataFrame, log_dir: Path, progress=None) -> pd.DataFrame:
    frames = []
    for i, build in enumerate(schedule.to_dict("records")):
        frames.append(run_build(cfg, build, tests, log_dir))
        if progress and ((i + 1) % 10 == 0 or i + 1 == len(schedule)):
            progress(f"  ran build {i + 1}/{len(schedule)}")
    return pd.concat(frames, ignore_index=True)
