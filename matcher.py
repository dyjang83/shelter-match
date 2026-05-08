"""
matcher.py
==========

Inference + explanation layer.

Given a trained model, a list of animals, and an adopter, produce a ranked
list of (animal, match_score, top_factors) tuples for the UI.

The "top factors" explanation works by perturbing each feature one at a
time and seeing how the predicted probability changes — a lightweight
local-importance method (think baby SHAP). It's not as rigorous as real
SHAP values but it's plenty for a demo and it has zero extra dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

from shelter import Adopter, Animal


# Categorical perturbation alternatives — used to compute "what if this
# feature changed?" for explanations.
PERTURBATION_ALTS = {
    "lives_in_apartment": [0, 1],
    "has_kids": [0, 1],
    "allergic_to_cats": [0, 1],
    "allergic_to_dogs": [0, 1],
    "prefers_quiet_home": [0, 1],
    "good_with_kids": [0, 1],
    "needs_quiet": [0, 1],
}


@dataclass
class MatchExplanation:
    feature: str
    contribution: float          # signed: positive = pushes toward "good match"
    plain_english: str           # human-readable factor for the UI


@dataclass
class MatchResult:
    animal: Animal
    score: float                 # probability in [0, 1]
    top_factors: List[MatchExplanation]


def _build_row(animal: Animal, adopter: Adopter) -> dict:
    """Combine animal + adopter into a single feature dict matching the
    schema the model was trained on.
    """
    return {**animal.feature_dict(), **adopter.feature_dict()}


def _explain(pipeline, base_row: dict, base_score: float, top_n: int = 3) -> List[MatchExplanation]:
    """Estimate each feature's contribution by counterfactual perturbation.

    For each feature, we flip it to its "opposite" (binary) or set it to
    a sensible alternative (numeric: clamp toward the dataset mean), then
    re-score. The change in score is the feature's local contribution —
    if removing/changing the feature would lower the score, the original
    value was helping.
    """
    contributions = []
    for feature, value in base_row.items():
        if feature in PERTURBATION_ALTS:
            alt_value = 1 - value if value in (0, 1) else value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            # Perturb toward 5 (midpoint of most numeric scales).
            alt_value = 5 if abs(value - 5) > 1 else value + 2
        else:
            continue  # categorical strings (species, size) — skip for now

        if alt_value == value:
            continue

        perturbed = dict(base_row)
        perturbed[feature] = alt_value
        perturbed_score = pipeline.predict_proba(pd.DataFrame([perturbed]))[0][1]
        delta = base_score - perturbed_score  # positive = original feature helped
        contributions.append((feature, delta, value, alt_value))

    contributions.sort(key=lambda t: abs(t[1]), reverse=True)

    explanations = []
    for feature, delta, value, _alt in contributions[:top_n]:
        explanations.append(MatchExplanation(
            feature=feature,
            contribution=float(delta),
            plain_english=_phrase(feature, value, delta),
        ))
    return explanations


def _phrase(feature: str, value, delta: float) -> str:
    """Turn a (feature, value, delta) triple into a friendly sentence."""
    direction = "boosts" if delta > 0 else "lowers"
    pretty = {
        "energy_level": f"animal energy level ({value}/10) {direction} the match",
        "activity_level": f"adopter activity level ({value}/10) {direction} the match",
        "hours_home_per_day": f"adopter is home {value}h/day — {direction} match",
        "lives_in_apartment": "adopter " + ("lives in an apartment" if value else "has a house") + f" — {direction} match",
        "has_kids": "adopter " + ("has kids" if value else "no kids") + f" — {direction} match",
        "needs_quiet": "animal " + ("needs quiet" if value else "tolerates noise") + f" — {direction} match",
        "prefers_quiet_home": "adopter " + ("prefers a quiet home" if value else "OK with activity") + f" — {direction} match",
        "good_with_kids": "animal " + ("good with kids" if value else "not great with kids") + f" — {direction} match",
        "allergic_to_cats": "adopter " + ("is allergic to cats" if value else "no cat allergy") + f" — {direction} match",
        "allergic_to_dogs": "adopter " + ("is allergic to dogs" if value else "no dog allergy") + f" — {direction} match",
        "experience_level": f"adopter experience ({value}/10) {direction} the match",
        "training_level": f"animal training level ({value}/10) {direction} the match",
        "age": f"animal age ({value}) {direction} the match",
    }
    return pretty.get(feature, f"{feature}={value} {direction} match")


def score_matches(pipeline, animals: List[Animal], adopter: Adopter) -> List[MatchResult]:
    """Score every animal for the given adopter, return ranked list."""
    results = []
    for animal in animals:
        row = _build_row(animal, adopter)
        score = float(pipeline.predict_proba(pd.DataFrame([row]))[0][1])
        explanations = _explain(pipeline, row, score)
        results.append(MatchResult(animal=animal, score=score, top_factors=explanations))

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def load_model(path: str = "models/match_model.joblib"):
    return joblib.load(path)


if __name__ == "__main__":
    # Quick test
    from shelter import Dog, Cat, Rabbit
    from data_gen import random_animal

    pipeline = load_model()
    animals = [random_animal() for _ in range(8)]
    adopter = Adopter(
        name="Test Adopter",
        lives_in_apartment=True,
        has_kids=False,
        hours_home_per_day=4,
        activity_level=3,
        experience_level=2,
        prefers_quiet_home=True,
    )

    results = score_matches(pipeline, animals, adopter)
    print(f"Adopter: apartment dweller, low activity, prefers quiet, home 4h/day\n")
    print(f"Ranked matches:\n")
    for r in results:
        print(f"  {r.score:.2f}  {r.animal.info()}")
        for f in r.top_factors:
            sign = "+" if f.contribution > 0 else "-"
            print(f"          {sign} {f.plain_english}")
        print()
