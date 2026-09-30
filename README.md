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
│   ├── 02_preprocessing.ipynb      # encoder/scaler grid + paired 
│   └── 03_cross_validation.ipynb      
├── results/                  # created automatically, one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    ├── diagnosis_log.json    # diagnosis log
    └── README.md             # problem description + data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Focus | Added to the pipeline |
|------|-------|-----------------------|
| 2 | Introduction & baseline pipeline | Project structure; single train/test split (no cross-validation); minimal preprocessing (drop rows with missing values, one-hot encode categoricals); logistic regression baseline; simple fairness check comparing our model's and COMPAS's false-positive rate by race; train-vs-test accuracy reporting; each run's report saved to `results/`. |
| 3 | EDA + preprocessing | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, duplicate check). `src/preprocessing.py` now handles leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, and the train/test split, replacing the old `dropna()`/`pd.get_dummies()`. Encoder/scaler (target encoding + standard scaling) chosen by an empirical grid over 15 repeated splits. Three redundant columns dropped (correlation + VIF). `config.yaml` gains `diagnostics` and `preprocessing` sections. |
| 4 | Preprocessing inside the pipeline + cross-validation -- evaluating a model honestly | A **locked final test set** (20%, stratified, seed 42) is set aside by `split_dev_test()` (replaces `split_train_test()`) and never scored; models are now judged by **stratified 5-fold cross-validation** of the whole pipeline (preprocessing + model) on the development set, reported per fold with mean ± std and the train-validation gap; the classification report and fairness check now use out-of-fold predictions; target encoding switched to scikit-learn's cross-fitting `TargetEncoder` (a row's own label never leaks into its own encoding), encoder/scaler set by hand in `config.yaml` (target encoding + robust scaling, reasons in the comments); **two fixes** in `clean_dataset()`: genuine `NaN`s in categorical columns were being turned into the string `"nan"` (a fake category), so 229 `c_charge_degree` gaps were never imputed or flagged -- fixed in `config.yaml` alone: `"nan"` added to `diagnostics.placeholder_tokens` (the category cleanup's last step turns listed tokens into `NaN`, after its text conversion); and it no longer drops rows -- de-duplication moved to a separate, training-only `drop_duplicate_rows()` (run before the dev/test split), so the same cleaning can run on new data where every row needs a prediction; `src/data_diagnostics.py` removed -- its one cleaning function (`flag_invalid_values`) moved into `preprocessing.py`, and the EDA-only checks (missingness test, duplicate counts) live in the EDA notebooks, not in every pipeline run; `dummy` (majority-class) model added as the floor to beat, and `random_forest` registered (sensible defaults, untuned); the final model is refit on the whole development set after CV; `config.yaml` gains `test_set` and `cv` sections -- see "Model evaluation" below |

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

**Encoder/scaler pair:** chosen by hand in `config.yaml` -- **target encoding** (compact, informative and **robust scaling** (median/IQR, so the few extreme counts don't set the scale). The alternatives (`onehot`/`ordinal`/`count`, `none`/`standard`/`minmax`) are one config change away.


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

This loads `config.yaml`, diagnoses and cleans the data, locks the final test set away, cross-validates preprocessing + model on the development set, and prints:
- **a per-fold cross-validation table** -- train and validation accuracy for each of the 5 folds, the gap between them, and their mean ± std. Comparing train and validation is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises.
- a classification report on the out-of-fold predictions
- a false-positive-rate-by-race comparison between our model and COMPAS's own score (same rows)
- the final model -- the same pipeline refit on all development rows (CV estimated how good it is; this is the model itself)
- a reminder of how many rows are in the locked test set -- which is **not** evaluated

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

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

The full weekly analysis is in [`result_analysis.md`](result_analysis.md).

