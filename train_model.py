"""
train_model.py
==============

Train the matching model on the synthetic dataset.

Approach
--------
1. Load synthetic data, split into train/test.
2. One-hot encode the categorical features (species, size).
3. Fit two models for comparison:
     - LogisticRegression (interpretable baseline)
     - GradientBoostingClassifier (better with non-linear interactions)
4. Report accuracy, ROC-AUC, and feature importances.
5. Persist the trained pipeline for the Streamlit app to load.

Why these two models? Logistic regression gives us a clean linear baseline
and easily interpretable coefficients. Gradient boosting handles the
interaction effects in the ground-truth function (e.g., apartment AND
large dog) that a linear model can't represent natively. Comparing them
in the demo is a nice "you can see why we picked the more complex model"
moment.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_COLS = ["species", "size"]
NUMERIC_COLS = [
    "age", "energy_level", "good_with_kids", "needs_quiet", "training_level",
    "lives_in_apartment", "has_kids", "hours_home_per_day", "activity_level",
    "experience_level", "allergic_to_cats", "allergic_to_dogs",
    "prefers_quiet_home",
]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def build_preprocessor() -> ColumnTransformer:
    """One-hot for categoricals, scale numerics. Wrapped in a ColumnTransformer
    so the same preprocessing applies at inference time inside the pipeline.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Train both models, return a dict of fitted pipelines + metrics."""
    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}

    for name, clf in [
        ("logistic_regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("gradient_boosting", GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )),
    ]:
        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("classifier", clf),
        ])
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}")
        print(f"ROC-AUC:  {auc:.3f}")
        print(classification_report(y_test, preds, target_names=["bad match", "good match"]))

        results[name] = {
            "pipeline": pipeline,
            "accuracy": acc,
            "auc": auc,
        }

    return results


def show_feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    """Pull feature importances out of the gradient boosting model.

    We need to reconstruct the post-encoding feature names because the
    one-hot encoder expands `species` into species_dog, species_cat, etc.
    """
    preprocess = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["classifier"]

    # Get expanded feature names from the ColumnTransformer.
    cat_names = preprocess.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS)
    feature_names = list(cat_names) + NUMERIC_COLS

    importances = classifier.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("\nTop 10 most important features (gradient boosting):")
    print(df.head(10).to_string(index=False))
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/synthetic_matches.csv")
    print(f"Loaded {len(df)} samples.")

    results = train_and_evaluate(df)

    # Save the better model (gradient boosting) for the app.
    best = results["gradient_boosting"]
    joblib.dump(best["pipeline"], "models/match_model.joblib")
    print(f"\nSaved gradient boosting model to models/match_model.joblib")

    # Persist feature importances for the UI to display.
    importances_df = show_feature_importances(best["pipeline"])
    importances_df.to_csv("models/feature_importances.csv", index=False)
