"""
Test-log writing, parsing and normalisation.

The framework writes one plain-text log file per build, in the style of a lab
automation log. Each test section contains:
  * fixed INFO lines (configuration, measured values),
  * WARN lines emitted by modem diagnostics when a reading looks abnormal
    (each is dropped with some probability, as real logging is inconsistent),
  * unrelated "nuisance" warnings that occur at random (instrument, network,
    disk) - the NLP model has to learn to ignore these,
  * an ERROR line stating which limit was violated, for failing tests.

The injected fault name never appears in the log. The triage models have to
infer it from symptoms, exactly like a validation engineer reading a log.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# (diagnostic key, threshold test, message template)
DIAGNOSTIC_WARNINGS = [
    ("sat_ratio", lambda v: v > 0.002,
     "[WARN] rx_frontend: ADC saturation on {pct:.2f}% of samples"),
    ("cpe_slope_rad", lambda v: abs(v) > 0.12,
     "[WARN] rx_track: pilot phase drifting {val:+.3f} rad/symbol"),
    ("evm_slope_db", lambda v: v > 0.12,
     "[WARN] rx_demod: EVM rising across frame ({val:+.2f} dB/symbol)"),
    ("image_corr", lambda v: v > 0.2,
     "[WARN] rx_demod: error correlated with mirror subcarrier (rho={val:.2f})"),
    ("sc_spread", lambda v: v > 2.2,
     "[WARN] rx_eq: per-subcarrier EVM spread high (std/mean={val:.2f})"),
    ("group_delay", lambda v: v < -0.7,
     "[WARN] rx_chest: negative channel group delay {val:+.2f} samples"),
    ("chan_min_db", lambda v: v < -15,
     "[WARN] rx_chest: deep spectral null {val:.1f} dB in channel estimate"),
]

NUISANCE_WARNINGS = [
    "[WARN] instrument: VSA response time {n} ms exceeds 100 ms",
    "[WARN] testbench: log buffer {n}% full, flushing",
    "[WARN] network: license server latency {n} ms",
    "[WARN] instrument: signal generator temperature drift {n} mK",
    "[WARN] scheduler: test slot started {n} s late",
    "[WARN] storage: results disk {n}% used",
]

WARN_DROP_PROBABILITY = 0.2       # chance a genuine diagnostic warning is not logged
NUISANCE_PROBABILITY = 0.35       # chance of each test getting a nuisance warning


def render_test_log(test: dict, build_id: str, result: dict, status: str, reasons: list,
                    rng: np.random.Generator, t0: datetime) -> str:
    """Produce the log text of one test case."""
    lines = []
    t = t0

    def emit(msg: str):
        nonlocal t
        t += timedelta(milliseconds=int(rng.integers(5, 400)))
        lines.append(f"{t.isoformat(timespec='milliseconds')} {msg}")

    lines.append(f"=== TEST {test['test_id']} BEGIN build={build_id} mod={test['modulation']} "
                 f"snr_db={test['snr_db']} channel={test['channel']} cfo={test['cfo']}")
    emit(f"[INFO] tb: configure signal generator mod={test['modulation']} power=-30dBm")
    emit(f"[INFO] tb: channel emulator profile={test['channel']} snr={test['snr_db']}dB cfo={test['cfo']}")
    emit("[INFO] rx_sync: preamble detected")
    emit(f"[INFO] rx_sync: cfo_est={result['cfo_est']:+.3f} subcarrier")
    emit(f"[INFO] rx_chest: group_delay={result['group_delay']:+.2f} samples")

    for key, test_fn, template in DIAGNOSTIC_WARNINGS:
        val = result[key]
        if test_fn(val) and rng.random() > WARN_DROP_PROBABILITY:
            emit(template.format(val=val, pct=100 * val))
    if rng.random() < NUISANCE_PROBABILITY:
        emit(NUISANCE_WARNINGS[int(rng.integers(len(NUISANCE_WARNINGS)))].format(n=int(rng.integers(50, 990))))

    emit(f"[INFO] result: evm={result['evm_pct']:.2f}% ber={result['ber']:.2e}")
    for r in reasons:
        emit(f"[ERROR] check failed: {r}")
    lines.append(f"=== TEST {test['test_id']} END status={status}")
    return "\n".join(lines)


_BEGIN = re.compile(r"^=== TEST (\S+) BEGIN build=(\S+) mod=(\S+) snr_db=(\S+) channel=(\S+) cfo=(\S+)")
_END = re.compile(r"^=== TEST (\S+) END status=(\S+)")


def parse_build_log(path: str | Path) -> list[dict]:
    """Split a build log into one record per test: params, status and the raw body text."""
    records, current = [], None
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            m = _BEGIN.match(line)
            if m:
                current = {"test_id": m.group(1), "build_id": m.group(2), "modulation": m.group(3),
                           "snr_db": float(m.group(4)), "channel": m.group(5), "cfo": float(m.group(6)),
                           "lines": []}
                continue
            m = _END.match(line)
            if m and current is not None:
                current["status"] = m.group(2)
                current["text"] = "\n".join(current.pop("lines"))
                records.append(current)
                current = None
                continue
            if current is not None:
                current["lines"].append(line)
    return records


_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\s*", re.M)
_NUMBER = re.compile(r"[+-]?\d+(\.\d+)?(e[+-]?\d+)?", re.I)


def normalize_log(text: str) -> str:
    """Standard log-mining pre-processing: drop timestamps, mask every number with <NUM>.

    This turns concrete lines into *templates* ("pilot phase drifting <NUM> rad/symbol"),
    so the model learns which kinds of messages occur, not the specific values.
    """
    text = _TIMESTAMP.sub("", text)
    text = _NUMBER.sub(" NUM ", text)
    return re.sub(r"\s+", " ", text).strip().lower()
