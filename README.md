# Baseline Predictive Pipeline -- ETAI

**Beatriz Marques** · 20231605 · forked from [sofiacper/ETAI-Pipeline](https://github.com/sofiacper/ETAI-Pipeline)

Predictive pipeline for two-year recidivism using ProPublica's COMPAS dataset -- the data behind a 2016 investigation into a risk-assessment algorithm used by US courts to inform bail and sentencing decisions. See `data/README.md` for the full problem description and data dictionary.

The pipeline started from the course baseline and is improved each week. Changes are logged in the progress table below.

## Project structure

```
├── main.py                   # entry point: runs the whole pipeline
├── config.yaml               # all tunable settings
├── requirements.txt
├── src/
│   ├── data.py               # data loading
│   ├── data_diagnostics.py   # missingness-mechanism test, domain-rule checks, duplicate check
│   ├── preprocessing.py      # leak-safe cleaning, preprocessing pipeline, train/test split
│   ├── model.py              # model construction
│   ├── evaluate.py           # accuracy + fairness check
│   └── results.py            # saves each run's report
├── notebooks/
│   ├── 01_eda_introduction.ipynb   # EDA: missingness, invalid values, duplicates, correlation + VIF
│   └── 02_preprocessing.ipynb      # encoder/scaler grid + paired comparison
├── results/                  # created automatically, one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    ├── diagnosis_log.json    # diagnosis log
    └── README.md             # problem description + data dictionary
```

## Pipeline progress

| Week | Focus | Added to the pipeline |
|------|-------|-----------------------|
| 2 | Introduction & baseline pipeline | Project structure; single train/test split (no cross-validation); minimal preprocessing (drop rows with missing values, one-hot encode categoricals); logistic regression baseline; simple fairness check comparing our model's and COMPAS's false-positive rate by race; train-vs-test accuracy reporting; each run's report saved to `results/`. |
| 3 | EDA + preprocessing | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, duplicate check). `src/preprocessing.py` now handles leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, and the train/test split, replacing the old `dropna()`/`pd.get_dummies()`. Encoder/scaler (target encoding + standard scaling) chosen by an empirical grid over 15 repeated splits. Three redundant columns dropped (correlation + VIF). `config.yaml` gains `diagnostics` and `preprocessing` sections. |

## Preprocessing decisions

Summary of the diagnosis in `01_eda_introduction.ipynb` and the empirical grid in `02_preprocessing.ipynb`.

| Column(s) | Issue found | Mechanism | What was done |
|---|---|---|---|
| `age` | 2.0% missing | MCAR | median impute, no indicator |
| `juv_fel_count` | 3.0% missing | MCAR | median impute, no indicator |
| `priors_count` | ~7% missing (incl. placeholder tokens) | MNAR -- tied to `age_cat` | median impute + `priors_count_was_missing` flag |
| `c_charge_degree` | 3.2% missing | MNAR -- tied to `age_cat` | mode impute + `c_charge_degree_was_missing` flag |
| `race` | ~1% missing (placeholder tokens) | MCAR | mode impute, no indicator (not a model feature) |
| `sex` | ~1.5% missing (incl. placeholder tokens) | MCAR | mode impute, no indicator |
| `age`, `decile_score`, `juv_fel_count`, `priors_count` | out-of-range or negative values | domain rule | converted to `NaN` before imputation |
| `sex` / `race` / `c_charge_degree` / `score_text` | inconsistent spelling (casing, whitespace, abbreviations) | data entry | canonicalized to one spelling per category |
| whole rows | 72 exact duplicates sharing a repeated `id` | data entry | dropped, kept first occurrence |
| `prior_offenses`, `age_in_months`, `juvenile_total` | redundant (r = 1.00, or an exact sum caught by VIF for `juvenile_total`) | multicollinearity | dropped |

**Encoder/scaler pair:** 4 encoders (one-hot, ordinal, count, target) × 4 scalers (none, standard, min-max, robust), scored by mean accuracy over 15 repeated train/test splits with logistic regression. **Target encoding + standard scaling won**, but a paired comparison against the runner-up showed the margin was within noise.

## Setup

Run once per machine:

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once, or use `venv\Scripts\activate.bat` in cmd.exe.

## Running the pipeline

With the environment active, from the project root:

```bash
python main.py
```

This loads `config.yaml`, diagnoses and cleans the data, trains the model, and prints:
- train and test accuracy side by side (to spot overfitting)
- a classification report on the test set
- a false-positive-rate-by-race comparison between the model and COMPAS's own score

Each run's report is also saved to `results/` as a timestamped file (e.g. `results/run_20260916_143012.txt`).

## Push to GitHub via Terminal

Standard workflow, from the project's root folder, with the venv active:
```bash
git add .
git commit -m "short description of what changed"
git push
```

**If `git push` asks for a password and rejects your normal GitHub password:** GitHub no longer accepts account passwords for git over HTTPS -- you need a **Personal Access Token (PAT)** instead.
1. On GitHub: **Settings -> Developer settings -> Personal access tokens -> Tokens (classic)** -> **Generate new token**, with at least `repo` scope.
2. When `git push` prompts for a password, paste the token instead (username stays your GitHub username).
3. So you're not asked every time: `git config --global credential.helper manager` (Windows, usually already set up by Git for Windows) or `git config --global credential.helper store` (caches it in plaintext -- fine on a personal machine, not a shared one).

Alternative: set up an SSH key once (`ssh-keygen -t ed25519`, then add the public key under **GitHub -> Settings -> SSH and GPG keys**) and use the repo's SSH remote URL (`git@github.com:...`) instead of HTTPS -- no token to manage or renew.

## Result Analysis

**Week 2 (16/09/2026) -- Logistic Regression vs. Decision Tree** *(baseline preprocessing)*

Logistic Regression performs better: higher **test accuracy** (67.7% vs. 66.8%) and a smaller **train-test gap** (0.001 vs. 0.012), so it generalises better and overfits less.

The Decision Tree has a slightly higher F1-score for **class 1** (0.64 vs. 0.63), but Logistic Regression is better overall and has a lower **false positive rate** for the largest race group, African-American (0.33 vs. 0.39).

**Week 3 (23/09/2026) -- After cleaning and preprocessing**

*TO-DO*