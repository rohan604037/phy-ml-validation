# PHYVAL — ML-Driven Validation Framework for an OFDM Baseband PHY

An automated validation framework for a wireless modem's physical layer, with machine learning on top:
automatic failure triage, NLP log classification, discovery of new bug types, intelligent test
selection, and build-level anomaly detection. It runs end to end with one command, in under a minute,
on a normal laptop.

> **OFDM** = Orthogonal Frequency-Division Multiplexing (the modulation used by Wi-Fi, LTE and 5G NR).
> **PHY** = Physical Layer. **ML** = Machine Learning. **NLP** = Natural Language Processing.
> All other abbreviations are defined in [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#glossary).

---

## What it does

```
 config/test_plan.yaml          (the specification: sweep, limits, channels, cost model)
          │
          ▼
 ┌──────────────────────┐   216 candidate tests, run 6x each on the bug-free "golden" design
 │ 1. Qualification     │── keeps the 119 tests that always pass
 └──────────────────────┘
          │
          ▼
 ┌──────────────────────┐   100 builds of the modem; 38 carry one of 6 injected hardware/firmware bugs
 │ 2. Regression        │── 11,900 test runs, 1,973 failures, one text log file per build
 └──────────────────────┘
          │
          ▼
 ┌──────────────────────┐   a. Auto-triage: Random Forest / Logistic Regression / TF-IDF NLP on logs
 │ 3. Machine learning  │   b. Discovery: K-Means clustering + open-set "unknown bug" detection
 └──────────────────────┘   c. Test selection: gradient boosting ranks tests for a new build
          │                 d. Anomaly detection: Isolation Forest on per-build telemetry
          ▼
 ┌──────────────────────┐
 │ 4. Report + email    │── self-contained HTML report + ready-to-send .eml file
 └──────────────────────┘
```

The device under test is a 64-subcarrier OFDM modem written from scratch in NumPy. It includes
preamble-based frequency-offset correction, channel estimation, equalisation, pilot phase tracking,
and BPSK, QPSK, 16-QAM and 64-QAM modulation. Six realistic bugs can be injected, each with a
severity from 0 to 1:

| Bug | Owning module | What goes wrong |
|---|---|---|
| `short_cp` | tx_framer | Cyclic prefix shorter than the channel's echo spread |
| `iq_imbalance` | tx_rf | I/Q gain/phase mismatch leaks energy onto mirror subcarriers |
| `adc_clipping` | rx_frontend | ADC full scale too low; signal peaks get clipped |
| `cfo_uncorrected` | rx_sync | Frequency offset only partly removed |
| `timing_offset` | rx_sync | FFT window starts late and picks up the next symbol |
| `stale_channel` | rx_chest | Equaliser uses a partly outdated channel estimate |

The bug name never appears in the logs or in the features. The models must infer it from symptoms,
the way a validation engineer would.

## Results (full run, `python -m phyval all`)

All models are evaluated only on builds they did not see during training.

| Capability | Result |
|---|---|
| Failure triage, best model (Random Forest on diagnostics) | **94.5% accuracy, 92.8% macro-F1**, grouped 5-fold cross-validation by build |
| Failure triage, log text only (TF-IDF + Logistic Regression) | 73.0% accuracy, 69.4% macro-F1 |
| Build-level root-cause call | **96%** of 74 failing builds (97% of genuinely buggy builds) |
| Test selection with 20% of test time | ML catches **33.4%** of failures vs 27.4% history heuristic vs 14.6% random; 100% of buggy builds detected (random: 66%) |
| Test selection with 30% of test time | ML **45.5%** vs history 37.6% vs random 23.6% |
| Anomaly detection (Isolation Forest) | 0 false alarms on 47 clean builds; 79% of buggy builds flagged; **3 of 4 "quiet" bugs** (pass rate ≥ 95%) caught |
| Unsupervised discovery | When a bug type is hidden from training, 57–89% of its failures are flagged "unknown" (vs 5–16% of known bugs) |

These numbers are reproducible: everything is seeded from `seed:` in the YAML file.
An honest reading of each result, including where the method falls short, is in
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#honest-limitations).

## Quick start

**Full beginner guide, including installing Python on Windows: [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md)**

```bash
# one-time setup (creates .venv, installs packages, runs the 33 unit tests)
./setup.sh                  # Windows: setup.bat

# run everything (about 45 s; add --quick for about 30 s)
./run.sh all                # Windows: run.bat all

# open the report
output/reports/regression_report.html
```

Validate a new build the way a validation team would:

```bash
./run.sh run-build --fault stale_channel --severity 0.5 --changed rx_chest --budget 0.3
```

```
ML test selection: 30 of 119 tests chosen for a 30% time budget (changed modules: rx_chest, telemetry)
NEW_20261010_041345: 0 passed, 30 failed
Predicted root cause: stale_channel (confidence 72%)
```

## Commands

| Command | What it does |
|---|---|
| `all [--quick] [--email-to X]` | Whole pipeline: qualify → regress → analyze → report |
| `qualify` / `regress` / `analyze` / `report` | Run one stage at a time |
| `run-build --fault F --severity S --changed M1,M2 --budget B` | Validate one new build: ML-select tests, run them, triage, report |
| `triage-log --log FILE` | Triage a raw log file with the NLP model only |
| `run-build --fault random --seed N` | A surprise build: bug chosen by the seed, revealed only in the report |
| `index` | Write `output/reports/index.html` linking every report (used for the website) |
| `faults` | List the injectable bugs |

Run any command with `--help` to see every option. Use `--out DIR` to write somewhere other than `output/`.

## Project layout

```
config/test_plan.yaml        the specification (edit this, not the code)
phyval/phy/ofdm.py           OFDM transmitter, channel, receiver, fault injection, diagnostics
phyval/framework/runner.py   test generation, qualification, build execution, regression history
phyval/framework/logs.py     log writing, log parsing, log normalisation for NLP
phyval/ml/triage.py          supervised root-cause classification (metrics + NLP), model selection
phyval/ml/discovery.py       K-Means clustering, open-set detection of unseen bugs
phyval/ml/selection.py       ML test selection under a time budget
phyval/ml/anomaly.py         Isolation Forest build-health monitoring
phyval/report/               HTML reports, PHY plots, email publication
phyval/__main__.py           command-line interface
tests/                       33 pytest unit tests
.github/workflows/           CI/CD: tests + pipeline on every push and nightly, live report website (GitHub Pages)
docs/                        how to run, how it works, resume and interview notes
```

## Technology

Python 3.9+, NumPy (signal processing), pandas (data handling), scikit-learn (Random Forest,
Logistic Regression, Histogram Gradient Boosting, Isolation Forest, K-Means, PCA, TF-IDF),
Matplotlib (plots), PyYAML (test specification), joblib (model persistence), pytest (unit tests),
GitHub Actions (CI/CD).
