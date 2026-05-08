"""
data_gen.py
===========

Generate synthetic (animal, adopter, was_good_match) training data.

Design philosophy
-----------------
The ML model never sees the rules below. We define a hidden ground-truth
compatibility function based on plausible domain knowledge, sample animals
and adopters from realistic distributions, then label each pair using the
hidden function plus noise. The model's job is to *recover* these patterns
from data — when feature importances line up with the rules, we know it
worked.

This is the same pattern used in research benchmarks: hand-design a
ground-truth function, generate labeled data, then evaluate whether a
learner can recover it. It also gives us a clean story to tell in the
demo — "the model learned that high-energy dogs in apartments are bad
matches, even though I never told it that."
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shelter import Adopter, Cat, Dog, Rabbit, Animal


# -----------------------------------------------------------------------------
# Random animal sampling
# -----------------------------------------------------------------------------
DOG_NAMES = ["Buddy", "Max", "Luna", "Bella", "Charlie", "Daisy", "Rocky", "Milo",
             "Lucy", "Cooper", "Bailey", "Sadie", "Tucker", "Zoe", "Duke", "Penny"]
CAT_NAMES = ["Whiskers", "Oliver", "Mittens", "Shadow", "Tigger", "Salem", "Pumpkin",
             "Felix", "Nala", "Simba", "Cleo", "Loki", "Mochi", "Biscuit", "Pepper"]
RABBIT_NAMES = ["Thumper", "Hazel", "Clover", "Cinnamon", "Pepper", "Snowball",
                "Bunbun", "Cocoa", "Marshmallow", "Pebbles"]


def random_dog() -> Dog:
    return Dog(
        name=random.choice(DOG_NAMES),
        age=random.randint(1, 12),
        energy_level=random.randint(3, 10),       # dogs skew high-energy
        size=random.choices(["small", "medium", "large"], weights=[2, 4, 3])[0],
        good_with_kids=random.random() < 0.7,
        needs_quiet=random.random() < 0.15,
        training_level=random.randint(1, 10),
    )


def random_cat() -> Cat:
    return Cat(
        name=random.choice(CAT_NAMES),
        age=random.randint(1, 15),
        energy_level=random.randint(1, 8),        # cats skew lower-energy
        size=random.choices(["small", "medium"], weights=[3, 2])[0],
        good_with_kids=random.random() < 0.5,
        needs_quiet=random.random() < 0.5,
        training_level=random.randint(1, 6),      # cats are less "trainable"
    )


def random_rabbit() -> Rabbit:
    return Rabbit(
        name=random.choice(RABBIT_NAMES),
        age=random.randint(1, 8),
        energy_level=random.randint(2, 7),
        size="small",
        good_with_kids=random.random() < 0.4,
        needs_quiet=True,                         # rabbits are skittish
        training_level=random.randint(1, 4),
    )


SAMPLERS = [random_dog, random_cat, random_rabbit]
SAMPLER_WEIGHTS = [0.5, 0.4, 0.1]                 # most shelter animals are dogs/cats


def random_animal() -> Animal:
    return random.choices(SAMPLERS, weights=SAMPLER_WEIGHTS)[0]()


# -----------------------------------------------------------------------------
# Random adopter sampling
# -----------------------------------------------------------------------------
ADOPTER_NAMES = ["Alice", "Bob", "Carla", "David", "Eva", "Frank", "Grace", "Henry",
                 "Iris", "Jamal", "Kira", "Liam", "Mia", "Noah", "Olivia", "Priya"]


def random_adopter() -> Adopter:
    return Adopter(
        name=random.choice(ADOPTER_NAMES),
        lives_in_apartment=random.random() < 0.45,
        has_kids=random.random() < 0.35,
        hours_home_per_day=random.randint(2, 16),
        activity_level=random.randint(1, 10),
        experience_level=random.randint(1, 10),
        allergic_to_cats=random.random() < 0.15,
        allergic_to_dogs=random.random() < 0.08,
        prefers_quiet_home=random.random() < 0.3,
    )


# -----------------------------------------------------------------------------
# Hidden ground-truth compatibility function
# -----------------------------------------------------------------------------
# This function returns a "true" compatibility score in [0, 1]. The training
# pipeline thresholds it (with noise) into a binary label. The model never
# sees this function — it has to learn the patterns from the labels.
#
# The rules below encode common-sense shelter wisdom. They are NOT secret
# from the user (in the demo we'll show what the model recovered vs. what
# we encoded), they're just hidden from the *model*.
# -----------------------------------------------------------------------------
def true_compatibility_score(animal: Animal, adopter: Adopter) -> float:
    """Compute the ground-truth match score in [0, 1].

    Higher = better match. The model's job is to learn this from labels
    without ever seeing this code.
    """
    score = 0.7  # neutral baseline

    # --- Hard incompatibilities (allergies) ---
    if animal.species == "cat" and adopter.allergic_to_cats:
        score -= 0.6
    if animal.species == "dog" and adopter.allergic_to_dogs:
        score -= 0.6

    # --- Energy compatibility ---
    # An active person matches with a high-energy animal; a sedentary
    # person matches with a low-energy animal. Mismatch hurts.
    energy_gap = abs(animal.energy_level - adopter.activity_level)
    score -= 0.04 * energy_gap

    # --- Apartment + large/high-energy dog = bad fit ---
    if adopter.lives_in_apartment and animal.species == "dog":
        if animal.size == "large":
            score -= 0.20
        if animal.energy_level >= 8:
            score -= 0.15

    # --- Kids in the home ---
    if adopter.has_kids and not animal.good_with_kids:
        score -= 0.30
    if adopter.has_kids and animal.species == "rabbit":
        # Rabbits are fragile; not ideal with small kids.
        score -= 0.15

    # --- Quiet preferences ---
    if animal.needs_quiet and not adopter.prefers_quiet_home:
        score -= 0.15
    if adopter.prefers_quiet_home and animal.energy_level >= 8:
        score -= 0.10

    # --- Time at home ---
    # Dogs need company; less time at home hurts dog matches more than cats.
    if animal.species == "dog" and adopter.hours_home_per_day < 5:
        score -= 0.20
    if animal.species == "cat" and adopter.hours_home_per_day < 3:
        score -= 0.05

    # --- Experience matters for hard-to-handle animals ---
    if animal.training_level <= 3 and adopter.experience_level <= 3:
        score -= 0.15  # untrained animal + first-time owner = trouble

    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


# -----------------------------------------------------------------------------
# Build a training dataframe
# -----------------------------------------------------------------------------
def build_dataset(n_samples: int = 2000, label_noise: float = 0.10,
                  seed: int = 42) -> pd.DataFrame:
    """Generate `n_samples` (animal, adopter, label) rows.

    Labels are binary: 1 if the pair is a good match, 0 otherwise.
    `label_noise` is the probability we flip the label, which forces the
    model to learn the underlying signal rather than memorize.
    """
    random.seed(seed)
    np.random.seed(seed)

    rows = []
    for _ in range(n_samples):
        animal = random_animal()
        adopter = random_adopter()
        true_score = true_compatibility_score(animal, adopter)

        # Threshold true score into a binary label, then add noise.
        label = 1 if true_score >= 0.55 else 0
        if random.random() < label_noise:
            label = 1 - label

        # Combine animal + adopter features into one flat row.
        row = {**animal.feature_dict(), **adopter.feature_dict(),
               "true_score": true_score, "label": label}
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = build_dataset(n_samples=2000)
    print(f"Generated {len(df)} samples.")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nFeature columns: {[c for c in df.columns if c not in ('label', 'true_score')]}")
    df.to_csv("data/synthetic_matches.csv", index=False)
    print("\nSaved to data/synthetic_matches.csv")
