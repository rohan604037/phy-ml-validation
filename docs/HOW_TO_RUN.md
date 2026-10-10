# How to run PHYVAL, step by step

This guide assumes no prior setup. Follow it once, and after that the project runs with a single
command whenever you want.

---

## Step 0: what you need

- A laptop running Windows 10/11, macOS or Linux
- About 500 MB of free disk space, for Python packages
- An internet connection for the first setup only

## Step 1: install Python (skip if `python --version` already shows 3.9 or newer)

**Windows**
1. Go to <https://www.python.org/downloads/> and download Python 3.12.
2. Run the installer. **Tick "Add python.exe to PATH"** on the first screen, then click *Install Now*.
3. Open **Command Prompt** (press Windows, type `cmd`, press Enter) and type:
   ```
   python --version
   ```
   You should see `Python 3.12.x`.

**macOS**: install from <https://www.python.org/downloads/>, or run `brew install python` if you use Homebrew.
Check with `python3 --version`.

**Linux (Ubuntu/Debian)**: `sudo apt install python3 python3-venv python3-pip`

## Step 2: put the project somewhere sensible

Unzip `phy-ml-validation.zip` into, for example, `Documents/phy-ml-validation`.
The folder should contain `README.md`, `setup.bat`, `setup.sh`, `phyval/`, `config/` and the other project files.

## Step 3: open a terminal inside the project folder

- **Windows**: open the folder in File Explorer, click the address bar, type `cmd` and press Enter.
  A Command Prompt opens already inside the folder.
- **macOS**: open Terminal, type `cd ` (with a trailing space), drag the folder onto the window and press Enter.
- **Linux**: right-click inside the folder and choose *Open in Terminal*.

## Step 4: one-time setup

| Windows | macOS / Linux |
|---|---|
| `setup.bat` | `./setup.sh` |

This does four things:
1. Creates a private Python environment in `.venv/`, so nothing is installed system-wide.
2. Installs NumPy, pandas, scikit-learn, Matplotlib, PyYAML, joblib and pytest.
3. Runs the 33 unit tests. You should see `33 passed`.
4. Prints `Setup complete.`

It takes 1–3 minutes. You only ever do this once.

## Step 5: run the whole pipeline

| Windows | macOS / Linux |
|---|---|
| `run.bat all` | `./run.sh all` |

Use `all --quick` for a shorter history (45 builds instead of 100, about 30 s instead of 45 s).

What the output means:

```
[1/4] Qualifying candidate tests against the golden design
  119 of 216 candidate tests qualified          <- tests the bug-free modem passes reliably
[2/4] Running regression history: 100 builds x 119 tests
  11,900 test executions, 1,973 failures        <- simulated build history with injected bugs
[3/4] Machine learning
 - auto-triage ...
  Random Forest (metrics)  accuracy=0.945 macro-F1=0.928   <- root-cause prediction on unseen builds
  selected model: Random Forest (metrics)                  <- best model is deployed automatically
 - intelligent test selection ...
  ml @ 20% time: catches 33.4% of failures ...             <- vs random 14.6%
 - build anomaly detection ...
[4/4] Building report
  report: .../output/reports/regression_report.html
```

## Step 6: open the report

Open `output/reports/regression_report.html` by double-clicking it. It opens in any browser,
works offline, and is a single file you can email or upload.

| Section | What to look at |
|---|---|
| Summary cards | Headline numbers |
| 1. Device under test | Constellation of every bug: what each one physically does to the signal |
| 2. Test automation | How the YAML specification turned into qualified tests |
| 3. Auto-triage | Model comparison, confusion matrix (which bugs get confused with which), feature importance |
| 4. Discovery | Clusters found without labels; how often an unseen bug is flagged "unknown" |
| 5. Test selection | Failures caught versus test time spent, for ML vs heuristic vs random |
| 6. Anomaly detection | Build-by-build anomaly score, with the quiet bugs circled |
| 7. Build history | Every build: bug, severity, changed modules, pass rate |

## Step 7: use it like a validation engineer (run it whenever you want)

Each command below needs `all` (Step 5) to have been run at least once, because it uses the trained models.

**Validate a new build, running only the tests the ML model picks for a 30% time budget**

```
run.bat run-build --fault cfo_uncorrected --severity 0.6 --changed rx_sync --budget 0.3
./run.sh run-build --fault cfo_uncorrected --severity 0.6 --changed rx_sync --budget 0.3
```

- `--fault`: which bug to inject. Use `none` for a clean build. `run.bat faults` lists them all.
- `--severity`: 0 to 1. Try small values such as 0.1 to see quiet bugs.
- `--changed`: the modules this build modified. The selection model uses them to choose tests.
- `--budget`: the fraction of full-suite test time to spend. `1.0` runs everything and also enables the anomaly check.
- `--seed 5`: makes a run exactly repeatable.

It prints the predicted root cause and writes:
- `output/reports/build_<id>.html`, the report for this build;
- `output/new_builds/<id>.log`, `<id>_results.csv` and `<id>_triage.csv`.

**Experiments worth trying**

| Try this | What you learn |
|---|---|
| `run-build --fault none` | A clean build; at most a rare noise-limited failure |
| `run-build --fault iq_imbalance --severity 0.1` | A quiet bug: very few failures, harder to triage |
| `run-build --fault stale_channel --changed telemetry --budget 0.2` | A misleading changed-module list: selection still finds failing tests, because historically fragile tests rank high. Compare with `--changed rx_chest` |
| `run-build --fault timing_offset --budget 1.0` | Full suite + anomaly check |
| `run-build --fault random --seed 123` | A surprise build: the bug is hidden until you open the report |

**Triage a raw log file using text only**

```
run.bat triage-log --log output\logs\B0042.log
./run.sh triage-log --log output/logs/B0042.log
```

Compare the answer with the `fault` column of `output/data/builds.csv`.

**Change what is tested**: edit `config/test_plan.yaml` (SNR points, limits, channels, number of
builds, bug probability, seed), then run `all` again. No code changes are needed.

**Keep several experiments side by side**: `run.bat --out results_v2 all` writes to `results_v2/`
instead of `output/`.

## Step 8: running without the helper scripts

The scripts only activate `.venv` and call `python -m phyval`. You can do the same by hand:

```
# Windows
.venv\Scripts\activate
python -m phyval all --quick
python -m pytest

# macOS / Linux
source .venv/bin/activate
python -m phyval all --quick
python -m pytest
```

## Step 9: put it on GitHub with automatic nightly runs and a live report website (CI/CD)

**A. Create the repository** on github.com: `+` → *New repository*, name `phy-ml-validation`, set it to
**Public**, and do not add a README. A free account can publish a website only from a public repository.

**B. Install Git** if `git --version` fails (<https://git-scm.com/download/win>, all defaults). Then set your
name once:
```
git config --global user.name "Rohan R"
git config --global user.email "rohanraghu610@gmail.com"
```

**C. Upload the project** from the project folder:
```
git init
git add .
git commit -m "PHY ML validation framework"
git branch -M main
git remote add origin https://github.com/rohan604037/phy-ml-validation.git
git push -u origin main
```
The first push opens a browser window to sign in to GitHub.

**D. Turn on the website (one time)**: repository → **Settings** → **Pages** → under *Build and deployment*,
set **Source** to **GitHub Actions**.

**E. Run it once now**: **Actions** tab → *phy-ml-regression* → **Run workflow** → **Run workflow**. Wait about
3–4 minutes for both jobs (*regression*, *publish-website*) to turn green.

**F. Open your live reports** at `https://rohan604037.github.io/phy-ml-validation/`. The index page links to
the regression report, tonight's nightly build report and the test-selection demo report.

### What happens automatically after that

| When | What runs | Result |
|---|---|---|
| Every push | Unit tests + full pipeline + reports | Website updated; a red cross on the commit if anything broke |
| Every night, 02:00 IST | The same, plus a **new random build** whose bug is chosen from the date | Website shows that night's build report |
| Anytime | *Run workflow* button | Same as the nightly run |

- **Website:** always shows the **latest** run; each run replaces the previous one.
- **History:** every run's reports and data are saved for 90 days. Open Actions → a run → *Artifacts* →
  `phyval-reports-<number>`.
- **What changes each night:** the regression history and its ML results are seeded, so they are identical
  every night. That is deliberate, because it proves the framework still reproduces its own results. What
  changes is the nightly build: a different bug (or no bug) and severity each date, which is then triaged blind.

### Things GitHub does that you should know

- **Pausing:** GitHub pauses scheduled runs in a public repository after **60 days without any commit**.
  Push any small change, or press *Enable workflow* in the Actions tab, to resume.
- **Timing:** scheduled runs can start a few minutes to an hour late when GitHub is busy.
- **Cost:** public repositories get Actions and Pages free.
- **Failure emails:** GitHub emails you if a run fails (Settings → Notifications → Actions).

`output/` and `.venv/` are listed in `.gitignore`, so generated files never get committed; the website is
built fresh on GitHub each run.

## Step 10 (optional): actually send the report by email

Every run writes an `.eml` file next to the report, which you can open in Outlook or Mail and send.
To send automatically, set these environment variables and add `--email-to`. The example uses Gmail,
which needs an *App Password* from your Google account's security settings.

```
# Windows (current Command Prompt only)
set PHYVAL_SMTP_HOST=smtp.gmail.com
set PHYVAL_SMTP_PORT=587
set PHYVAL_SMTP_USER=you@gmail.com
set PHYVAL_SMTP_PASSWORD=your-app-password
run.bat report --email-to you@gmail.com

# macOS / Linux
export PHYVAL_SMTP_HOST=smtp.gmail.com PHYVAL_SMTP_PORT=587 PHYVAL_SMTP_USER=you@gmail.com PHYVAL_SMTP_PASSWORD=your-app-password
./run.sh report --email-to you@gmail.com
```

Never put the password in a file that goes to GitHub.

## Where everything goes

| Path | Contents |
|---|---|
| `output/data/test_plan_qualified.csv` | All 216 candidate tests, golden EVM and whether each one qualified |
| `output/data/builds.csv` | The build history: injected bug, severity, changed modules |
| `output/data/runs.csv` | Every test execution with all measurements (open in Excel) |
| `output/logs/B0000.log` … | Raw text log of every build |
| `output/data/*_results.json` | All ML metrics |
| `output/models/*.joblib` | Trained models, reused by `run-build` and `triage-log` |
| `output/plots/*.png` | Every figure |
| `output/reports/*.html`, `*.eml` | Reports and email drafts |
| `output/new_builds/` | Artifacts of `run-build` |

## Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` (Windows) | Python was installed without "Add to PATH". Re-run the installer, choose *Modify*, and tick it |
| `./setup.sh: Permission denied` | Run `chmod +x setup.sh run.sh` once |
| `Run setup.bat first` | The `.venv` folder is missing; run the setup step |
| `Missing output/models/triage.joblib` | Run `all` (or `all --quick`) before `run-build` or `triage-log` |
| pip install fails behind a college proxy | Use a different network, or `pip install --proxy http://host:port -r requirements.txt` |
| Numbers differ slightly from the README | Different library versions can change Random Forest internals; trends stay the same |
