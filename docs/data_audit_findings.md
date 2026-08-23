# Data Audit Findings — Heart Disease Dataset (P1 / Milestone 5-6)

## Source
Mirrored from `sharmaroshan/Heart-UCI-Dataset` (raw.githubusercontent.com), a
widely-used derivative of the UCI Cleveland Heart Disease processed dataset,
matching the popular Kaggle "heart-disease-uci" version (303 rows, 14
columns, target pre-binarized). Original UCI source: UCI Machine Learning
Repository, "Heart Disease" (id: 45), CC BY 4.0.

## Raw shape
303 rows × 14 columns. No `NaN` values reported by pandas — but see below,
this is misleading.

## Findings

### 1. Duplicate row
One exact duplicate row pair found (index 163/164 in the raw file: age 38,
male, all fields identical). **Action: dropped one copy.**

### 2. Disguised missing values in `ca` and `thal`
The UCI codebook defines:
- `ca` (number of major vessels colored by fluoroscopy): valid range 0–3
- `thal`: valid range 1–3 (in some encodings), or 0–2 in this 0-indexed mirror

This file contains **5 rows with `ca == 4`** and **2 rows with `thal == 0`**,
both outside the documented valid range. In the original UCI `.data` file
these are `?` (explicit missing-value markers); this mirror appears to have
coerced them to an out-of-range integer instead of `NaN`, which is why
pandas' `isnull()` reports zero missing values — the missingness is hidden,
not absent. **Action: dropped 6 affected rows** (after de-duplication) rather
than imputing, since they're a small fraction (~2%) of an already-small
dataset and imputing categorical placeholders here would add complexity out
of proportion to the benefit for a capstone-scale project.

### 3. Target polarity — requires your confirmation
The original UCI codebook states the `target`/`goal` field is 0 = disease
absent, 1–4 = disease present (collapsed to 1 for binary tasks).

However, checking correlation direction between `target` and features with
an unambiguous clinical interpretation in **this specific file**:

| Feature | Clinical meaning of "high" value | Correlation with target==1 in this file |
|---|---|---|
| `exang` (exercise-induced angina) | Worse | **-0.44** (target==1 → less angina) |
| `oldpeak` (ST depression) | Worse | **-0.43** (target==1 → less depression) |
| `ca` (vessels blocked) | Worse | **-0.39** (target==1 → fewer blocked) |
| `thalach` (max heart rate achieved) | Better (reduced chronotropic response is a known CAD marker) | **+0.42** (target==1 → higher, i.e. healthier) |

All four point the same direction: in this file, `target == 1` behaves like
the **healthier** group, not the diseased group — the opposite of the
official codebook description. This matches a documented community
discussion around this exact Kaggle mirror (`ronitf/heart-disease-uci`,
discussion #105877) reporting the same inversion.

**Decision applied in `preprocessing.py` (default `flip_target=True`):**
the target column is flipped so that, for the rest of this project,
`target == 1` means **heart disease present**. This is implemented as an
explicit, documented, overridable parameter — not a silent transformation.

**This is flagged for your sign-off** — if you have independent access to
the original UCI `.data` file (or the `ucimlrepo` package) and can confirm
the polarity directly, we should verify against that rather than relying on
correlation-direction inference before this goes in the final report.

### 4. Class balance
54.5% / 45.5% in the raw file (before the target flip, direction unaffected
by flip). This is comfortably balanced — no resampling (SMOTE etc.) needed
for this dataset, unlike some of the alternative disease datasets discussed
in Section 6 of the architecture document.

### 5. Feature types for the preprocessing pipeline
- **Nominal categorical** (need one-hot encoding, not ordinal treatment):
  `cp`, `restecg`, `slope`, `thal`. Pearson correlation coefficients against
  these were *not* used to infer clinical direction for this reason — their
  integer codes don't represent an ordered scale.
- **True binary**: `sex`, `fbs`, `exang`.
- **Continuous numeric** (candidates for scaling in the Logistic Regression
  pipeline only — tree models don't need it): `age`, `trestbps`, `chol`,
  `thalach`, `oldpeak`, `ca`.

## Final processed shape
296 rows after de-duplication and invalid-row removal → 236 train / 60 test
(stratified 80/20 split, `random_state=42`), positive-class rate preserved
within 1 percentage point of the full-dataset rate in both splits.
