# Resume entry and interview preparation

## Before you add this to your resume

1. Run it yourself (`run.bat all`) and read every section of the report.
2. Read `phyval/phy/ofdm.py` and `phyval/ml/triage.py` line by line until you can explain them without notes.
3. Make at least one change of your own. Good options:
   - add a seventh bug, such as `phase_noise`;
   - add a PyTorch MLP (Multi-Layer Perceptron) to the triage comparison;
   - add a 256-QAM modulation and limit.
4. Push it to your GitHub and link it.

Interviewers will ask about any line on the resume. Claim only what you can defend.

---

## Resume entry (numbers are from the full run; re-check them after your own run)

**ML-Driven Validation Framework for an OFDM Baseband PHY** | Python, NumPy, scikit-learn, Pandas, pytest, GitHub Actions
- Built an OFDM baseband PHY model with CFO estimation, LS channel estimation and pilot tracking,
  plus 6 injectable RF/firmware faults, and a YAML-driven automation framework that qualifies a
  216-test plan against a golden design and runs a 100-build regression (11,900 test executions).
- Developed ML auto-triage that classifies root cause from receiver diagnostics. It reaches 92.8%
  macro-F1 on unseen builds (grouped cross-validation) and 96% build-level accuracy, and adds a
  TF-IDF NLP classifier on normalised test logs.
- Implemented gradient-boosting test selection that catches 2.3× more failures than random at 20%
  of test time, and an Isolation Forest build monitor with 0 false alarms that flagged 3 of 4 bugs
  invisible to pass/fail limits. Integrated it into GitHub Actions CI with auto-generated HTML/email reports.

Abbreviations on the resume itself can stay as written: interviewers in this area know them. If the
resume has room, write "Orthogonal Frequency-Division Multiplexing (OFDM)" once.

Also update the profile line for this application, for example:
> "...seeking roles in wireless validation, test automation and ML-for-test..."

---

## Likely interview questions, with answers grounded in this project

**1. Why simulate a modem instead of using real data?**
Real cellular validation data is proprietary. A simulation gives ground-truth labels for every
failure, so triage can be trained and evaluated honestly. The pipeline is data-agnostic: replacing
`simulate_link` with real instrument measurements would leave the ML layer unchanged.

**2. How did you avoid data leakage?**
Three ways:
- Triage uses **GroupKFold by build**, because failures from the same build share a root cause and
  look alike. A random split inflated accuracy.
- Test selection uses a **chronological split** and computes historical failure rates **only from
  earlier builds**.
- The bug name never appears in the logs or the features; a unit test checks the logs.

**3. Why macro-F1 instead of accuracy?**
Classes are imbalanced: timing_offset produces about 700 failures, adc_clipping about 90. A model
could get high accuracy by being good only at the big classes. Macro-F1 weights every class equally.

**4. Which features matter most, and does that make physical sense?**
Image correlation (I/Q imbalance), ADC saturation ratio (clipping), pilot phase drift (residual
CFO) and channel group delay (late timing window) dominate, as the Random Forest importance plot in
the report shows. Each corresponds to the physical mechanism of one bug.

**5. Where does triage fail?**
short_cp, stale_channel and timing_offset get confused, because all three produce
inter-symbol-interference-like errors on multipath channels. A fix would be more targeted
diagnostics, for example error versus delay tap, or per-symbol error structure.

**6. Your NLP model is much worse than the metric model. Why keep it?**
Logs carry less information: warnings are thresholded, some are dropped, and numbers are masked.
Often, though, logs are all you have from a remote test station; `triage-log` handles that case.
Combining the two by averaging probabilities didn't help, so model selection deploys the best
single model. I report that result rather than hiding it.

**7. Why mask numbers in logs?**
Without masking, "drift 0.301" and "drift 0.298" become different tokens and the vocabulary blows
up with near-unique values. Masking turns lines into templates, so the model learns which *kinds*
of message appear. This is standard in log mining (for example, Drain-style template extraction).

**8. How does test selection decide what to run?**
A gradient-boosting model predicts P(fail) per test from the test's parameters, the build's changed
modules and the test's past failure rate. Tests are ranked by P(fail) ÷ duration, i.e. expected
failures per second, and taken greedily until the time budget is used. That is a greedy
approximation to the knapsack problem.

**9. The history heuristic detects all buggy builds with 5% of time and yours needs 15%. Isn't the heuristic better?**
It depends on the goal. For "is this build broken at all?", running the historically fragile tests
is a strong baseline, because most bugs break many tests. For "find as many distinct failures as
possible" (which is what triage and debugging need), ML is better: 45% vs 38% of failures at 30%
time. In practice I'd combine them: run a small fragile-test smoke set first, then ML-ranked tests.

**10. How does the anomaly detector work, and why train only on baseline builds?**
Each build is summarised by pass rates and median diagnostics. An Isolation Forest learns what
known-good builds look like and scores how isolated each new build is. Training on mixed builds
would teach it that bugs are normal. Its value is quiet bugs: builds with a pass rate of 95% or more
that pass/fail checks would accept. It caught 3 of 4.

**11. What are the limitations?**
- The data is simulated, with one bug per build and diagnostics I designed myself, so real-world
  accuracy will be lower.
- The history is small, about 6 builds per bug type.
- There is no channel coding, MIMO or 5G NR numerology.

The full list is in HOW_IT_WORKS.md.

**12. How would you deploy this for a real validation team?**
- Replace the simulator with an instrument-control layer, for example PyVISA driving a vector signal
  analyser and signal generator.
- Store results in a database instead of CSV, and run in CI on every firmware drop.
- Retrain models nightly.
- Route low-confidence triage calls to engineers and feed their labels back as training data
  (active learning).
- Publish reports to SharePoint via the Microsoft Graph API.

**13. Explain CFO estimation.**
The short training symbol uses only even subcarriers, so its time-domain signal repeats every 32
samples. A frequency offset ε rotates the second half relative to the first by 2π·ε·32/64 = π·ε.
So ε = angle(Σ conj(first half) × second half) / π, which is unambiguous for |ε| < 1 subcarrier spacing.

**14. Why does a stale channel estimate do nothing on an AWGN channel?**
On a flat channel the frequency response is the same constant for every realisation. An "outdated"
estimate is therefore identical to the current one. A unit test checks this.

**15. Why does a short CP only hurt the VEH_A channel?**
The CP absorbs echoes shorter than itself. PED_A's longest echo is 3 samples, and even the shortest
buggy CP is 4 samples, so nothing changes. VEH_A has an echo at 13 samples, which exceeds a 4–12
sample CP and causes inter-symbol interference.
