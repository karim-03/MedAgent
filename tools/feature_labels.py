"""
tools/feature_labels.py
Single source of truth for "raw ML field name -> human-readable label",
shared by tools/risk_explanation.py (SHAP display) and
tools/patient_intake.py (follow-up acknowledgment text). Previously
duplicated as a private dict inside risk_explanation.py — pulled out once
a second tool needed the same mapping, rather than importing another
module's private name or re-typing the list.
"""

FEATURE_LABELS = {
    "age": "age",
    "sex": "sex",
    "cp": "chest pain type",
    "trestbps": "resting blood pressure",
    "chol": "cholesterol",
    "fbs": "fasting blood sugar",
    "restecg": "resting ECG result",
    "thalach": "maximum heart rate achieved",
    "exang": "exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "ST segment slope",
    "ca": "number of blocked vessels",
    "thal": "thalassemia result",
}
