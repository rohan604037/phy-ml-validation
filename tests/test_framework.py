"""Unit tests for the automation framework, logs and ML helpers."""
from pathlib import Path

import numpy as np
import pandas as pd

from phyval.config import load_config
from phyval.framework.logs import normalize_log, parse_build_log
from phyval.framework.runner import (check_limits, generate_candidates, make_build_schedule, run_build)
from phyval.ml.selection import select_tests

CFG = load_config()


def test_candidates_cover_the_full_sweep():
    c = generate_candidates(CFG)
    sw = CFG["sweep"]
    assert len(c) == len(sw["modulation"]) * len(sw["snr_db"]) * len(sw["channel"]) * len(sw["cfo"])
    assert c["test_id"].is_unique
    assert (c["cost_s"] > 0).all()


def test_limit_checks():
    assert check_limits(CFG, "QAM64", 5.0, 0.0) == []
    assert len(check_limits(CFG, "QAM64", 9.0, 0.0)) == 1
    assert len(check_limits(CFG, "QAM64", 9.0, 0.5)) == 2


def test_schedule_baseline_is_clean_and_bugs_touch_their_module():
    s = make_build_schedule(CFG, 60)
    base = s[s["baseline"]]
    assert (base["fault"] == "none").all()
    for _, b in s[s["fault"] != "none"].iterrows():
        assert CFG["fault_module"][b["fault"]] in b["changed_modules"].split(",")


def test_build_log_roundtrip(tmp_path: Path):
    tests = generate_candidates(CFG).head(8)
    build = {"build_num": 1, "build_id": "BTEST", "date": "2026-01-01", "fault": "timing_offset",
             "severity": 0.8, "changed_modules": "rx_sync"}
    runs = run_build(CFG, build, tests, tmp_path)
    recs = parse_build_log(tmp_path / "BTEST.log")
    assert [r["test_id"] for r in recs] == list(runs["test_id"])
    assert [r["status"] for r in recs] == list(runs["status"])
    # The injected fault name must never leak into the log text.
    assert all("timing_offset" not in r["text"] for r in recs)


def test_log_normalisation_masks_numbers_and_timestamps():
    t = "2026-01-01T02:00:00.123 [WARN] rx_track: pilot phase drifting +0.301 rad/symbol"
    n = normalize_log(t)
    assert "2026" not in n and "0.301" not in n and "num" in n


def test_greedy_selection_respects_budget():
    scores = np.array([0.9, 0.1, 0.5, 0.8])
    costs = np.array([10.0, 10.0, 10.0, 40.0])
    idx = select_tests(scores, costs, 25.0)
    assert costs[idx].sum() <= 25.0
    assert set(idx) == {0, 2}
