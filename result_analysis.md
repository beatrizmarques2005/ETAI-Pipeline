# Result Analysis

## How results are evaluated

`python main.py` evaluates one train/test split (`random_state=42`). A single split can favour
one model by chance, so each comparison below also reports the **mean ± standard deviation over
15 stratified 80/20 splits** (`random_state` 0–14), with each split identical across all configurations.

---

## Weekly Experiments
Accuracy is the mean ± std over 15 stratified 80/20 splits, unless stated otherwise.

### Summary
---

| Week | Main change | Best model | Mean test accuracy |
|---|---|---|---:|
| 2 | Baseline pipeline | Logistic Regression | 0.672 ± 0.011 |
| 3 | EDA-driven preprocessing | Decision Tree (`max_depth=5`) | 0.675 ± 0.011 |


### Week 2 — Baseline
---

**What changed:** rows with missing values dropped, `pd.get_dummies()` on text columns.

| Model | Mean test accuracy |
|---|---:|
| Logistic Regression | **0.672 ± 0.011** |
| Decision Tree (`max_depth=5`) | 0.662 ± 0.014 |

**Findings:** `dropna()` removed 14% of the rows, mostly older defendants. Race labels were not cleaned, so the fairness check split the same group into several spellings.

### Week 3 — EDA and preprocessing
---

**What changed:** category cleanup, domain-rule checks, duplicate removal, median/mode imputation with MNAR flags, redundant columns dropped, target encoding and standard scaling.

| Model | Mean test accuracy | vs Week 2 |
|---|---:|---:|
| Logistic Regression | 0.670 ± 0.014 | −0.002 |
| Decision Tree (`max_depth=5`) | **0.675 ± 0.011** | +0.013 |

**Challenge answers:**

1. **Other imputers:** ...
2. **New data issue:** 6 rows had an `age_cat` that contradicted `age`. These are now recomputed from `age`. 10 `score_text`/`decile_score` mismatches were flagged; neither column is a model feature.
3. **Week 2 vs Week 3:** preprocessing improved the decision tree and left logistic regression about the same. False-positive rates fell for both models, mainly because more rows were kept and categories were cleaned. The single `random_state=42` run looked worse, but its test sets differ in size (1,252 vs 1,443) and are not directly comparable.

**Best model:** Decision Tree (`max_depth=5`). Logistic regression is close and has lower false-positive rates.


<!-- TEMPLATE
### Week N — <topic>
---

**What changed:** ...

| Model | Mean test accuracy | vs Week N-1 |
|---|---:|---:|
| ... | ... | ... |

**Findings:** ...

**Best model:** ...
-->
