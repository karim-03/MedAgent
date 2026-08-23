# Model Evaluation & Selection Findings — P1 (Milestones 7-8)

## Setup
- 4 candidate models: Logistic Regression, Decision Tree, Random Forest, XGBoost.
- Each tuned via `GridSearchCV` (5-fold stratified CV, scoring = ROC-AUC) on
  the 236-row training set only.
- Each evaluated exactly once on the untouched 60-row test set after tuning.

## Results

| Model | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8948 | 0.8667 | 0.8846 | 0.8214 | 0.8519 | **0.9576** |
| Decision Tree | 0.7570 | 0.8500 | 0.8519 | 0.8214 | 0.8364 | 0.8856 |
| **Random Forest** | 0.8866 | 0.8833 | 0.8889 | **0.8571** | **0.8727** | 0.9509 |
| XGBoost | 0.8826 | 0.8833 | 0.9200 | 0.8214 | 0.8679 | 0.9464 |

Confusion matrix (Random Forest, test set): 29 true negatives, 3 false
positives, 4 false negatives, 24 true positives.

## Selection: Random Forest

Logistic Regression technically has the single highest test ROC-AUC
(0.9576 vs. 0.9509), but on a 60-row test set that ~0.7-point gap is not a
meaningful difference — a couple of different patients landing on either
side of the decision boundary would close it. Decision Tree is clearly
worse across every metric and cross-validation is unstable (0.757 CV
ROC-AUC vs. ~0.88-0.89 for the others) — expected, since a single tree is
higher-variance than an ensemble on a 236-row training set.

Between the three closely-matched models (Logistic Regression, Random
Forest, XGBoost), Random Forest has the best recall (0.857) and F1 (0.873).
Per the architecture doc's decision to weight recall alongside ROC-AUC for a
clinical screening context (a missed positive is costlier than an extra
follow-up test), **Random Forest is selected** — implemented as an explicit
rule in `select_best_model()`: among models within 0.02 ROC-AUC of the best,
pick the highest-recall one. This is a rule you can defend in the report
methodology section, not a manual eyeball call.

**Honest caveat for the report's limitations section**: n=60 test set means
every single prediction is worth ~1.7 percentage points of any metric. None
of the differences between Logistic Regression / Random Forest / XGBoost
here should be reported as statistically significant without a proper
confidence interval (e.g. bootstrap resampling of the test set, or repeated
stratified CV) — worth adding as a stretch goal if time allows, and worth
stating plainly either way rather than implying more precision than 60
samples can support.

## Feature importance (Random Forest, impurity-based) and SHAP agree

Top features by both impurity-based importance and SHAP: `thal` (thalassemia
defect type), `ca` (vessels blocked), `oldpeak` (ST depression), `exang`
(exercise-induced angina), `thalach` (max heart rate achieved). This
ordering matches established cardiology risk factors for coronary artery
disease, which is a useful cross-check: it's independent evidence that the
target-polarity flip applied in preprocessing (see
`data_audit_findings.md`) was the correct call — if the label had still
been inverted, these features would have shown up with implausible
(reversed) importance directions.

## Artifacts produced
- `ml/models/best_model.joblib` — fitted Random Forest pipeline (preprocessing + classifier bundled together, so it's inference-ready as-is).
- `ml/models/model_comparison.csv` — full comparison table.
- `outputs/figures/roc_curves_all_models.png`
- `outputs/figures/confusion_matrix_random_forest.png`
- `outputs/figures/feature_importance_random_forest.png`
- `outputs/figures/shap_importance_random_forest.png`

## Next milestone
P2 — Knowledge Base: source, chunk, and prepare the offline medical
reference documents for the RAG layer.
