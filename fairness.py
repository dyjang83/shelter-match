"""
fairness.py
===========

Fairness audit: does the model systematically disadvantage certain
animals across all adopter types?

This isn't just academic — animal shelters worry constantly about
"hard to place" residents (older, larger, lower-trained). If a matching
algorithm consistently ranks them low across ALL adopter profiles, it
amplifies an existing bias rather than helping shelters address it.

Methodology
-----------
1. Generate a representative sample of adopter profiles (covering the
   feature space, not just the training distribution).
2. For each animal under audit, score it against every adopter and
   record the average match probability and the rank distribution.
3. Compare protected groups: e.g., young vs. senior animals, well-trained
   vs. poorly-trained, small vs. large.
4. Report disparity metrics: differences in mean score, in median rank,
   and in the share of "Strong match" predictions (>= 0.6).

We use *demographic parity* as the high-level fairness lens: across a
neutral pool of adopters, do groups receive comparable opportunity to
be ranked highly? Equalized odds would require ground-truth labels per
animal, which we don't have at inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shelter import Adopter, Animal
from data_gen import random_adopter, random_animal
from train_model import FEATURE_COLS


# -----------------------------------------------------------------------------
# Audit input
# -----------------------------------------------------------------------------
def representative_adopters(n: int = 200, seed: int = 7) -> list[Adopter]:
    """Sample adopters spanning the feature space.

    We use the same generator as training data but with a different seed,
    giving us out-of-sample evaluation profiles.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    return [random_adopter() for _ in range(n)]


def audit_pool(n_per_group: int = 30, seed: int = 11) -> list[Animal]:
    """Sample animals to be audited, balanced across species."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    animals = []
    from data_gen import random_dog, random_cat, random_rabbit
    for _ in range(n_per_group):
        animals.append(random_dog())
        animals.append(random_cat())
        animals.append(random_rabbit())
    return animals


# -----------------------------------------------------------------------------
# Score a single animal against an adopter pool
# -----------------------------------------------------------------------------
def _animal_pool_scores(pipeline, animal: Animal, adopters: list[Adopter]) -> np.ndarray:
    rows = [{**animal.feature_dict(), **a.feature_dict()} for a in adopters]
    df = pd.DataFrame(rows)[FEATURE_COLS]
    return pipeline.predict_proba(df)[:, 1]


def audit_animals(pipeline, animals: list[Animal],
                  adopters: list[Adopter]) -> pd.DataFrame:
    """For each audited animal, compute summary stats over the adopter pool."""
    rows = []
    for animal in animals:
        scores = _animal_pool_scores(pipeline, animal, adopters)
        rows.append({
            "name": animal.name,
            "species": animal.species,
            "age": animal.age,
            "energy_level": animal.energy_level,
            "size": animal.size,
            "training_level": animal.training_level,
            "needs_quiet": animal.needs_quiet,
            "good_with_kids": animal.good_with_kids,
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "p90_score": float(np.percentile(scores, 90)),
            "share_strong_match": float(np.mean(scores >= 0.6)),
            "share_zero_chance": float(np.mean(scores < 0.20)),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Group disparity reports
# -----------------------------------------------------------------------------
def disparity_by_group(audit_df: pd.DataFrame, group_col: str,
                       group_fn=None) -> pd.DataFrame:
    """Compare mean/median scores across groups defined by `group_col`.

    `group_fn` optionally bucketizes a continuous column (e.g. age into
    "young", "adult", "senior").
    """
    df = audit_df.copy()
    if group_fn is not None:
        df["group"] = df[group_col].apply(group_fn)
    else:
        df["group"] = df[group_col]

    summary = df.groupby("group").agg(
        n=("mean_score", "size"),
        mean_score=("mean_score", "mean"),
        median_score=("median_score", "mean"),
        share_strong_match=("share_strong_match", "mean"),
        share_zero_chance=("share_zero_chance", "mean"),
    ).round(3).reset_index()
    return summary


def age_bucket(age: int) -> str:
    if age <= 2: return "young (<=2)"
    if age <= 7: return "adult (3-7)"
    return "senior (8+)"


def training_bucket(level: int) -> str:
    if level <= 3: return "low (1-3)"
    if level <= 6: return "medium (4-6)"
    return "high (7-10)"


# -----------------------------------------------------------------------------
# Disparity metric: ratio of strong-match share between groups
# -----------------------------------------------------------------------------
def disparity_ratios(group_summary: pd.DataFrame) -> dict:
    """Compute pairwise disparities for the share of strong matches.

    A disparity ratio of 1.0 means parity. A ratio of 0.5 means the
    disadvantaged group gets strong matches half as often as the
    advantaged group.
    """
    groups = group_summary["group"].tolist()
    shares = dict(zip(groups, group_summary["share_strong_match"]))
    pairs = {}
    for g1 in groups:
        for g2 in groups:
            if g1 >= g2:
                continue
            denom = max(shares[g1], shares[g2])
            num = min(shares[g1], shares[g2])
            ratio = num / denom if denom > 0 else float("nan")
            disadvantaged = g1 if shares[g1] < shares[g2] else g2
            pairs[f"{g1} vs {g2}"] = {
                "ratio": round(ratio, 3),
                "disadvantaged": disadvantaged,
            }
    return pairs


# -----------------------------------------------------------------------------
# Main audit driver
# -----------------------------------------------------------------------------
def run_audit(pipeline, n_animals_per_group: int = 30, n_adopters: int = 200) -> dict:
    animals = audit_pool(n_per_group=n_animals_per_group)
    adopters = representative_adopters(n=n_adopters)

    audit_df = audit_animals(pipeline, animals, adopters)

    by_species = disparity_by_group(audit_df, "species")
    by_age = disparity_by_group(audit_df, "age", group_fn=age_bucket)
    by_training = disparity_by_group(audit_df, "training_level", group_fn=training_bucket)
    by_size = disparity_by_group(audit_df, "size")

    return {
        "audit_df": audit_df,
        "by_species": by_species,
        "by_age": by_age,
        "by_training": by_training,
        "by_size": by_size,
        "species_disparity": disparity_ratios(by_species),
        "age_disparity": disparity_ratios(by_age),
        "training_disparity": disparity_ratios(by_training),
        "size_disparity": disparity_ratios(by_size),
    }


if __name__ == "__main__":
    import joblib
    pipeline = joblib.load("models/match_model.joblib")
    results = run_audit(pipeline)

    print("=== Per-species ===")
    print(results["by_species"].to_string(index=False))
    print("\n=== Per-age ===")
    print(results["by_age"].to_string(index=False))
    print("\n=== Per-training ===")
    print(results["by_training"].to_string(index=False))
    print("\n=== Per-size ===")
    print(results["by_size"].to_string(index=False))

    print("\n=== Disparity ratios (lower = more disparity) ===")
    print("By species:", results["species_disparity"])
    print("By age:", results["age_disparity"])
    print("By training:", results["training_disparity"])
    print("By size:", results["size_disparity"])

    results["audit_df"].to_csv("reports/fairness_audit.csv", index=False)
    print("\nSaved per-animal audit to reports/fairness_audit.csv")
