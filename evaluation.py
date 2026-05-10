"""
evaluation.py
=============

Rigorous model evaluation beyond a single accuracy number.

Provides:
  - Stratified k-fold cross-validation (so the headline metrics aren't
    a lucky train/test split).
  - Calibration analysis (does a 70% predicted score correspond to ~70%
    actual good matches?). Critical for a recommendation system where
    we surface scores directly to users.
  - Per-species confusion matrices (does the model under-perform on
    rabbits because they're underrepresented?).
  - Failure-mode breakdown: what kinds of (animal, adopter) pairs does
    the model get wrong?
  - Baseline comparison across logistic regression, gradient boosting,
    XGBoost, and a simple MLP.

All functions return structured results suitable for plotting in the
notebook rather than printing — the notebook controls presentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

import xgboost as xgb

from train_model import build_preprocessor, FEATURE_COLS


# -----------------------------------------------------------------------------
# Cross-validation
# -----------------------------------------------------------------------------
@dataclass
class CVResults:
    """Per-fold metrics, plus mean and std for headline reporting."""
    accuracy: List[float] = field(default_factory=list)
    auc: List[float] = field(default_factory=list)
    f1: List[float] = field(default_factory=list)
    brier: List[float] = field(default_factory=list)
    log_loss: List[float] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "accuracy_mean": float(np.mean(self.accuracy)),
            "accuracy_std": float(np.std(self.accuracy)),
            "auc_mean": float(np.mean(self.auc)),
            "auc_std": float(np.std(self.auc)),
            "f1_mean": float(np.mean(self.f1)),
            "f1_std": float(np.std(self.f1)),
            "brier_mean": float(np.mean(self.brier)),
            "log_loss_mean": float(np.mean(self.log_loss)),
        }


def cross_validate_model(model_factory, df: pd.DataFrame, n_splits: int = 5,
                         seed: int = 42) -> CVResults:
    """Run stratified k-fold CV, returning per-fold metrics.

    `model_factory` is a zero-arg callable that returns a fresh unfitted
    pipeline. We use a factory rather than a single pipeline so each fold
    gets a clean model (no leakage from prior fits).
    """
    X = df[FEATURE_COLS]
    y = df["label"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = CVResults()

    for train_idx, test_idx in skf.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        pipeline = model_factory()
        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        results.accuracy.append(accuracy_score(y_test, preds))
        results.auc.append(roc_auc_score(y_test, proba))
        results.f1.append(f1_score(y_test, preds))
        results.brier.append(brier_score_loss(y_test, proba))
        results.log_loss.append(log_loss(y_test, proba))

    return results


# -----------------------------------------------------------------------------
# Model factories — wrap each model in build_preprocessor for fair comparison
# -----------------------------------------------------------------------------
def make_logistic() -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def make_gradient_boosting() -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )),
    ])


def make_xgboost() -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=42, verbosity=0,
        )),
    ])


def make_mlp() -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=500, random_state=42
        )),
    ])


MODEL_FACTORIES = {
    "Logistic Regression": make_logistic,
    "Gradient Boosting": make_gradient_boosting,
    "XGBoost": make_xgboost,
    "MLP (32, 16)": make_mlp,
}


def compare_baselines(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Cross-validate every model and return a clean comparison table."""
    rows = []
    for name, factory in MODEL_FACTORIES.items():
        cv = cross_validate_model(factory, df, n_splits=n_splits)
        s = cv.summary()
        rows.append({
            "Model": name,
            "Accuracy": f"{s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f}",
            "ROC-AUC": f"{s['auc_mean']:.3f} ± {s['auc_std']:.3f}",
            "F1": f"{s['f1_mean']:.3f} ± {s['f1_std']:.3f}",
            "Brier (lower=better)": f"{s['brier_mean']:.3f}",
            "Log loss (lower=better)": f"{s['log_loss_mean']:.3f}",
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Calibration analysis
# -----------------------------------------------------------------------------
def calibration_data(pipeline: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray,
                     n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Return (predicted_probs, actual_fractions) for plotting a reliability diagram.

    A perfectly calibrated model produces points on the diagonal: when it
    predicts 70%, 70% of those cases are actually positive. Most ML models
    out-of-the-box are NOT well calibrated, so this is a real check.
    """
    proba = pipeline.predict_proba(X_test)[:, 1]
    fraction_positive, mean_predicted = calibration_curve(
        y_test, proba, n_bins=n_bins, strategy="quantile"
    )
    return mean_predicted, fraction_positive


# -----------------------------------------------------------------------------
# Per-species breakdown
# -----------------------------------------------------------------------------
def per_species_metrics(pipeline: Pipeline, df_test: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy and AUC separately for dogs, cats, rabbits.

    A model with strong overall accuracy can still be much weaker on a
    minority class — important to surface for any deployment decision.
    """
    rows = []
    X = df_test[FEATURE_COLS]
    proba = pipeline.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    df_test = df_test.copy()
    df_test["pred"] = preds
    df_test["proba"] = proba

    for species in df_test["species"].unique():
        sub = df_test[df_test["species"] == species]
        if len(sub) == 0 or sub["label"].nunique() < 2:
            continue
        rows.append({
            "Species": species,
            "n": len(sub),
            "Positive rate": f"{sub['label'].mean():.2f}",
            "Accuracy": f"{accuracy_score(sub['label'], sub['pred']):.3f}",
            "AUC": f"{roc_auc_score(sub['label'], sub['proba']):.3f}",
            "F1": f"{f1_score(sub['label'], sub['pred'], zero_division=0):.3f}",
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Failure mode analysis
# -----------------------------------------------------------------------------
def failure_modes(pipeline: Pipeline, df_test: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Find the most-confident wrong predictions.

    These are the cases the model is sure about but wrong — usually the
    most informative for understanding what the model has failed to learn.
    """
    X = df_test[FEATURE_COLS]
    proba = pipeline.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    df = df_test.copy()
    df["proba"] = proba
    df["pred"] = preds
    df["correct"] = df["pred"] == df["label"]
    df["confidence"] = np.abs(proba - 0.5)  # distance from decision boundary

    wrong = df[~df["correct"]].sort_values("confidence", ascending=False)
    return wrong.head(top_n)[
        ["species", "age", "energy_level", "size", "lives_in_apartment",
         "has_kids", "activity_level", "hours_home_per_day",
         "label", "pred", "proba"]
    ]


if __name__ == "__main__":
    df = pd.read_csv("data/synthetic_matches.csv")
    print("=== Baseline comparison (5-fold CV) ===")
    comparison = compare_baselines(df)
    print(comparison.to_string(index=False))
    comparison.to_csv("reports/baseline_comparison.csv", index=False)
    print("\nSaved to reports/baseline_comparison.csv")
