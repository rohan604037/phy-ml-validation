# How PHYVAL works

This document explains every part of the project and the reasoning behind each design decision.
Read it before an interview: almost every question about this project is answered here.

---

## 1. The device under test: an OFDM baseband PHY (`phyval/phy/ofdm.py`)

### Why simulate a modem at all?
Real cellular validation data is proprietary. A simulated modem with *physically correct* behaviour
gives labelled data: for every failure, we know the true root cause. That is exactly what is needed
to train and fairly evaluate triage models.

### Numerology (similar to IEEE 802.11a/g Wi-Fi)
| Parameter | Value |
|---|---|
| FFT size | 64 subcarriers |
| Used subcarriers | 52 (48 data + 4 pilots); DC and band edges left empty |
| Cyclic prefix (CP) | 16 samples |
| Modulations | BPSK, QPSK, 16-QAM, 64-QAM, Gray-coded, unit average power |
| Frame | STS + LTS preamble, then 20 data symbols |

### Transmitter
1. Random bits are mapped to constellation symbols.
2. Symbols are placed on data subcarriers and pilots on subcarriers ±7 and ±21.
3. An IFFT converts each symbol to the time domain, and the last 16 samples are copied to the front as the CP.

### Channel
- **Multipath**: Rayleigh-fading taps with the ITU Pedestrian-A or Vehicular-A delay profiles,
  scaled to this sample rate, or a flat AWGN channel. Each test uses a fixed channel realisation,
  like a recorded channel-emulator profile, so tests are repeatable and only the noise varies.
- **CFO**: the signal is multiplied by e<sup>j2π·ε·n/N</sup>, with ε given as a fraction of the subcarrier spacing.
- **Noise**: complex Gaussian noise at the test's SNR.

### Receiver
1. **AGC and ADC**: the signal is normalised to unit RMS and clipped at the ADC full-scale level (4× RMS when bug-free).
2. **CFO estimation**: the STS uses only even subcarriers, so it repeats every 32 samples. The
   phase of the correlation between the two halves gives ε = angle/π. Unit tests confirm the error
   stays below 0.02 over ±0.45.
3. **Channel estimation**: least squares, Ĥ[k] = Y[k] / X<sub>LTS</sub>[k].
4. **Equalisation**: zero-forcing, Z[k] = Y[k] / Ĥ[k].
5. **Pilot tracking**: the common phase error of each symbol is measured on the pilots and removed.
6. **Demodulation**: hard decision to the nearest constellation point.

### Diagnostics: what the models learn from
A real modem exposes counters and estimates to firmware logs. Ours computes:

| Diagnostic | Meaning | Bug it usually points to |
|---|---|---|
| `sat_ratio` | Fraction of ADC samples at full scale | adc_clipping |
| `cpe_slope_rad` | Drift of pilot phase per symbol | cfo_uncorrected (residual frequency offset) |
| `image_corr` | Correlation of error on subcarrier k with the symbol on −k | iq_imbalance (mirror-image leakage) |
| `group_delay` | Phase slope of the channel estimate, in samples | timing_offset (late FFT window gives a negative delay) |
| `evm_slope_db` | EVM growth across the frame | timing/ISI effects |
| `sc_spread` | How unevenly error is spread across subcarriers | stale_channel, short_cp (frequency-selective errors) |
| `chan_min_db` | Deepest null in the channel estimate | Distinguishes genuine fading from bugs |
| `pilot_evm_db` | Error measured on the pilots | General link quality |

### The six bugs, and why they are realistic
Each bug only hurts under certain conditions, just as real bugs do:

| Bug | When it actually breaks a test |
|---|---|
| `short_cp` | Only in VEH_A, the channel whose echoes (13 samples) exceed the shortened CP |
| `stale_channel` | Only on multipath channels; on a flat channel every estimate looks the same |
| `cfo_uncorrected` | Only when the test applies a frequency offset |
| `iq_imbalance`, `adc_clipping` | Mostly on dense constellations (16/64-QAM) with tight EVM limits |
| `timing_offset` | Almost everywhere; a global bug |

These interactions are what make test selection and triage genuine learning problems. If every bug
broke every test, there would be nothing to learn.

## 2. Spec-driven automation (`phyval/framework/`)

- **Specification to tests**: `config/test_plan.yaml` defines a sweep of 4 modulations × 6 SNRs ×
  3 channels × 3 CFOs, giving 216 candidates. Changing the plan needs no code change.
- **Qualification**: each candidate is run 6 times on the golden (bug-free) design and kept only if
  it always passes, which leaves 119 tests. Without this step, the suite would be full of tests that
  fail because the spec is unreachable at that SNR, not because of bugs. This mirrors how a validation
  team qualifies a new test before adding it to regression.
- **Limits**: EVM per modulation (BPSK 25%, QPSK 17.5%, 16-QAM 12.5%, 64-QAM 8%, in the style of
  3GPP LTE requirements) and BER ≤ 10⁻³.
- **Cost model**: each test has an estimated lab duration (channel-emulator setup, signal-generator
  retune). This is what test selection saves.
- **Build history**:
  - 100 builds. The first 15 are known-good baselines.
  - Afterwards, each build has a 45% chance of carrying one bug, with severity uniform between 0.05 and 1.
  - Every build lists the modules it changed. A buggy build always includes the module that owns the
    bug, plus some unrelated ones as noise, as in real change lists.
- **Logs**: one text log per build. Diagnostic warnings appear when a reading is abnormal, but each
  is randomly dropped 20% of the time, and unrelated nuisance warnings (instrument latency, disk usage)
  are mixed in. The bug name is never written. A unit test enforces this.

## 3. Machine learning (`phyval/ml/`)

### 3a. Auto-triage (`triage.py`): supervised classification
- **Input**: one failing test, with its diagnostics and its log text.
- **Output**: one of 6 bugs, or `noise_limited` (a failure on a clean build).
- **Models compared**:
  - Logistic Regression and Random Forest on the diagnostic features;
  - TF-IDF over unigrams and bigrams of the *normalised* log text, followed by Logistic Regression;
  - a soft-vote combination of the two.
- **Log normalisation**: timestamps are removed and every number is masked as `NUM`, so
  "drifting +0.301 rad/symbol" and "drifting −0.12 rad/symbol" become the same template. This is
  standard log-mining practice. Without it, the vocabulary is mostly unique numbers and the model overfits.
- **Evaluation: GroupKFold by build (5 folds).**
  - All failures of a build share one root cause and look alike. With a random row split, the model
    would see near-identical siblings in training, and accuracy would be inflated.
  - Grouping by build means every score reflects performance on *builds the model has never seen*.
- **Model selection**: the model with the best cross-validated macro-F1 is deployed automatically.
  Macro-F1 is used rather than accuracy because classes are imbalanced: timing_offset produces far
  more failures than adc_clipping.
- **Build-level call**: per-failure probabilities are summed across the build, and the top class is
  reported with its share of the evidence. Below 60%, the tool says "route to an engineer".

**Result**:
- Random Forest: 94.5% accuracy, 92.8% macro-F1.
- Build-level call: 96% correct.
- Remaining confusion is between short_cp, stale_channel and timing_offset, which all produce
  inter-symbol-interference-like errors. This is physically reasonable.

### 3b. Discovery (`discovery.py`): unsupervised learning
- **K-Means** (k = number of classes) on standardised symptom features, with no labels.
  - Agreement with the true causes: ARI 0.35, NMI 0.51, purity 63%.
  - Clustering alone recovers the structure only partially. That is the expected result: labels add a lot.
- **Open-set detection**: each bug is hidden from training in turn, then its failures are shown to the model.
  - 57–89% of those failures get top-class probability below 0.6 and are flagged "unknown".
  - Only 5–16% of known-bug failures are flagged.
  - So the classifier is usually uncertain about bugs it has never seen, rather than confidently wrong.

### 3c. Test selection (`selection.py`)
- **Goal**: given a new build's changed modules, rank tests by the probability that they fail.
- **Model**: Histogram Gradient Boosting.
- **Features**: test parameters, changed-module flags, and the test's historical failure rate.
  The history is computed only from earlier builds, so no future information leaks in.
- **Split**: chronological. Train on builds 0–69, test on builds 70–99, as it would be used in production.
- **Ranking**: P(fail) ÷ cost, i.e. expected failures caught per second of lab time. This is a greedy
  answer to a knapsack problem under a time budget.
- **Baselines**: random order, and "historically flaky tests first".

**Result**: ROC-AUC 0.74 for predicting failing tests. At 20% of test time:

| Strategy | Failures caught | Buggy builds detected |
|---|---|---|
| ML | 33.4% | 100% |
| History heuristic | 27.4% | 100% |
| Random | 14.6% | 66% |

### 3d. Anomaly detection (`anomaly.py`)
- **Idea**: limits catch bugs only once they break a limit, so look instead for builds whose overall
  telemetry is unusual.
- **Build summary vector**: pass rates, median EVM per modulation, and medians of each diagnostic.
- **Model**: an Isolation Forest trained *only* on the 15 baseline builds. Its threshold is the highest
  baseline score, which means zero false alarms on baseline by construction.

**Result**:
- ROC-AUC 1.00: every buggy build scores above every clean build.
- At the conservative threshold: 0 false alarms on 47 clean builds, 79% recall.
- 3 of 4 quiet bugs flagged. A quiet bug is a buggy build whose pass rate is still at least 95%,
  which pass/fail checks would treat as healthy.

## 4. Reporting and CI/CD

- **HTML report** (`report/html.py`): one self-contained file with every image embedded, dark-mode
  aware, ready to email or upload to SharePoint.
- **Email** (`report/notify.py`): always writes an `.eml` file. It sends via SMTP when `PHYVAL_SMTP_*`
  environment variables are set. Credentials are never stored in code.
- **GitHub Actions** (`.github/workflows/regression.yml`): runs on every push, pull request, manual
  trigger and nightly schedule. It runs the unit tests, the pipeline and a demo build, then uploads
  the reports as artifacts.
- **Unit tests** (`tests/`, 33 tests):
  - Gray coding; modulation round-trip; CFO estimator accuracy.
  - Every bug degrades EVM *and* moves its signature diagnostic; bugs are harmless where physics says they should be.
  - Log round-trip, with no label leakage; log normalisation; the time-budget constraint.

## Honest limitations

Say these before an interviewer finds them.

1. **Simulated, not real, data.**
   - The modem is a simplified OFDM link: no channel coding, no MIMO, no 5G NR numerology, and no
     real RF impairments beyond the six modelled.
   - The framework and ML methods transfer to real data; the specific accuracies do not.
2. **The bugs are cleaner than real bugs.**
   - Each buggy build has exactly one bug, and the diagnostics were designed by the same person who
     designed the bugs.
   - Real triage has multiple simultaneous issues, setup problems and flaky instruments, so expect
     lower accuracy on real data.
3. **The log NLP model is weaker than the metric model (69% vs 93% macro-F1)**, and combining them
   did not help. This is reported rather than hidden. Logs carry less information than the raw
   diagnostics, since warnings are thresholded and sometimes dropped. The NLP model is still useful
   when only logs are available (`triage-log`).
4. **Test selection's advantage is moderate.**
   - The history heuristic reaches 100% build detection with only 5% of test time; ML needs 15%.
     Many bugs break many tests, so "run the historically fragile tests" is already a strong baseline.
   - ML wins clearly on catching *individual* failures: 45% vs 38% at 30% time.
5. **The anomaly threshold is conservative.**
   - ROC-AUC is 1.0, yet recall is 79%. A lower threshold would catch more bugs, at the risk of false
     alarms that the 15-build baseline cannot estimate well.
   - With `--budget < 1`, the anomaly check is skipped, because summaries of a partial suite are not
     comparable to the baseline.
6. **Small history.** There are only about 6 buggy builds per bug type, so cross-validation
   estimates have wide uncertainty (roughly ±3–5 points).

## Mapping to the Apple job description

| Job description asks for | Where it is in this project |
|---|---|
| Automation frameworks and test infrastructure for PHY validation | `framework/`, YAML-driven test generation, qualification, logs |
| Translate specifications into automated test strategies; identify validation gaps | YAML test plan → qualified suite; qualification shows which spec points are unreachable |
| AI/ML auto-triage tools; report generation and publication | `ml/triage.py`, `report/html.py`, `report/notify.py` |
| ML models that predict failure patterns and optimise coverage | `ml/selection.py`, `ml/anomaly.py` |
| Intelligent test selection to reduce execution time | `ml/selection.py` with a time-budget knapsack |
| NLP to parse logs, categorise failures, generate insights | `framework/logs.py` parser and normaliser + TF-IDF classifier, `triage-log` command |
| Predictive models from historical data and telemetry | Isolation Forest on build telemetry; history features in selection |
| CI/CD integration | `.github/workflows/regression.yml` |
| Python, scikit-learn, Pandas, Matplotlib, Git | Throughout |
| Wireless communications / signal processing | `phy/ofdm.py` |
| Anomaly detection, model evaluation, unsupervised learning | `anomaly.py`, grouped cross-validation, `discovery.py` |

## Glossary

| Term | Meaning |
|---|---|
| ADC | Analog-to-Digital Converter |
| AGC | Automatic Gain Control |
| ARI | Adjusted Rand Index (cluster vs label agreement, 1 = perfect, 0 = random) |
| AWGN | Additive White Gaussian Noise |
| BER | Bit Error Rate |
| BPSK / QPSK | Binary / Quadrature Phase-Shift Keying |
| CFO | Carrier Frequency Offset |
| CI/CD | Continuous Integration / Continuous Delivery |
| CP | Cyclic Prefix |
| CPE | Common Phase Error (phase rotation shared by all subcarriers of one symbol) |
| DC | Direct Current (the zero-frequency subcarrier, left empty) |
| EVM | Error Vector Magnitude |
| F1 / macro-F1 | Harmonic mean of precision and recall / its average over classes |
| FFT / IFFT | Fast Fourier Transform / Inverse FFT |
| I/Q | In-phase / Quadrature |
| ISI | Inter-Symbol Interference |
| ITU | International Telecommunication Union |
| LS | Least Squares |
| LTS / STS | Long / Short Training Symbol |
| MIMO | Multiple-Input Multiple-Output |
| NLP | Natural Language Processing |
| NMI | Normalised Mutual Information |
| NR | New Radio (5G) |
| OFDM | Orthogonal Frequency-Division Multiplexing |
| PCA | Principal Component Analysis |
| PHY | Physical Layer |
| QAM | Quadrature Amplitude Modulation |
| RF | Radio Frequency (also used for Random Forest in the ML sections) |
| RMS | Root Mean Square |
| ROC-AUC | Area Under the Receiver Operating Characteristic curve |
| SMTP | Simple Mail Transfer Protocol |
| SNR | Signal-to-Noise Ratio |
| TF-IDF | Term Frequency – Inverse Document Frequency |
| 3GPP | 3rd Generation Partnership Project (the cellular standards body) |
