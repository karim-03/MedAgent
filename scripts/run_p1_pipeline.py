"""One-shot driver: train, compare, evaluate, explain. Not part of the
permanent package layout — a convenience script for this milestone's run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

from ml.training.train import run, load_processed_data
from ml.evaluation.metrics import plot_confusion_matrix, plot_roc_curves, plot_feature_importance
from ml.evaluation.shap_analysis import run_shap_summary

results_df, fitted_pipelines, best_name = run(save=True)

X_train, X_test, y_train, y_test = load_processed_data()
best_pipe = fitted_pipelines[best_name]
y_pred = best_pipe.predict(X_test)

cm = plot_confusion_matrix(y_test, y_pred, best_name)
print("\nConfusion matrix:\n", cm)

plot_roc_curves(fitted_pipelines, X_test, y_test)

imp_df = plot_feature_importance(best_pipe, best_name)
print("\nTop features:\n", imp_df.to_string(index=False))

run_shap_summary(best_pipe, X_test, best_name)

print("\nAll figures saved to outputs/figures/")
