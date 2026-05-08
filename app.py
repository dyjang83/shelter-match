"""
app.py
======

Streamlit web app for the ML-powered animal shelter.

Pages
-----
1. Shelter — view current residents, intake new animals.
2. Find a Match — fill out an adopter profile, get ranked matches with
   per-animal explanations.
3. Adopt — confirm an adoption (updates the recursive history chain).
4. History — look up any animal's full adoption chain.
5. About the Model — show feature importances and metrics, so the demo
   tells the full story.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from shelter import Adopter, Shelter, make_animal
from data_gen import random_animal
from matcher import load_model, score_matches


# -----------------------------------------------------------------------------
# Session state — Streamlit reruns the script on every interaction, so we
# keep persistent state in st.session_state.
# -----------------------------------------------------------------------------
def init_state():
    if "shelter" not in st.session_state:
        st.session_state.shelter = Shelter()
        # Seed with a few animals so the app isn't empty on first load.
        for _ in range(6):
            st.session_state.shelter.intake(random_animal())

    if "model" not in st.session_state:
        st.session_state.model = load_model()

    if "last_matches" not in st.session_state:
        st.session_state.last_matches = []

    if "last_adopter" not in st.session_state:
        st.session_state.last_adopter = None


def page_shelter():
    st.header("Shelter Residents")
    shelter = st.session_state.shelter

    residents = shelter.residents
    if not residents:
        st.info("No animals currently in the shelter.")
    else:
        rows = []
        for a in residents:
            rows.append({
                "Name": a.name,
                "Species": a.species,
                "Age": a.age,
                "Energy": f"{a.energy_level}/10",
                "Size": a.size,
                "Good with kids": "yes" if a.good_with_kids else "no",
                "Needs quiet": "yes" if a.needs_quiet else "no",
                "Training": f"{a.training_level}/10",
                "Past adoptions": a.history.depth() if a.history else 0,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Intake a new animal")

    with st.form("intake_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Name", value="")
            species = st.selectbox("Species", ["dog", "cat", "rabbit"])
        with col2:
            age = st.number_input("Age (years)", min_value=0, max_value=25, value=3)
            size = st.selectbox("Size", ["small", "medium", "large"])
            energy = st.slider("Energy level", 1, 10, 5)
        with col3:
            training = st.slider("Training level", 1, 10, 5)
            good_with_kids = st.checkbox("Good with kids", value=True)
            needs_quiet = st.checkbox("Needs a quiet home", value=False)

        submitted = st.form_submit_button("Add to shelter")
        if submitted:
            if not name.strip():
                st.error("Please enter a name.")
            else:
                animal = make_animal(
                    species,
                    name=name.strip(),
                    age=int(age),
                    energy_level=int(energy),
                    size=size,
                    good_with_kids=good_with_kids,
                    needs_quiet=needs_quiet,
                    training_level=int(training),
                )
                msg = shelter.intake(animal)
                st.success(msg)
                st.rerun()


def page_match():
    st.header("Find a Match")
    st.write(
        "Fill out the adopter's profile. The trained gradient boosting model "
        "will rank every available animal and explain why each one is a "
        "better or worse fit."
    )

    with st.form("adopter_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Adopter name", value="Alex")
            apartment = st.checkbox("Lives in an apartment", value=False)
            kids = st.checkbox("Has kids at home", value=False)
            quiet = st.checkbox("Prefers a quiet home", value=False)
            cat_allergy = st.checkbox("Allergic to cats", value=False)
            dog_allergy = st.checkbox("Allergic to dogs", value=False)
        with col2:
            hours = st.slider("Hours home per day", 0, 24, 8)
            activity = st.slider("Activity level", 1, 10, 5)
            experience = st.slider("Pet ownership experience", 1, 10, 5)

        submitted = st.form_submit_button("Find matches")

    if submitted:
        adopter = Adopter(
            name=name.strip() or "Adopter",
            lives_in_apartment=apartment,
            has_kids=kids,
            hours_home_per_day=int(hours),
            activity_level=int(activity),
            experience_level=int(experience),
            allergic_to_cats=cat_allergy,
            allergic_to_dogs=dog_allergy,
            prefers_quiet_home=quiet,
        )
        residents = st.session_state.shelter.residents
        if not residents:
            st.warning("No animals in the shelter to match against.")
            return
        matches = score_matches(st.session_state.model, residents, adopter)
        st.session_state.last_matches = matches
        st.session_state.last_adopter = adopter

    if st.session_state.last_matches:
        st.subheader(f"Ranked matches for {st.session_state.last_adopter.name}")
        for r in st.session_state.last_matches:
            with st.container(border=True):
                top_col, _ = st.columns([3, 1])
                with top_col:
                    pct = int(round(r.score * 100))
                    label = "Strong match" if r.score >= 0.6 else (
                        "Worth considering" if r.score >= 0.35 else "Probably not"
                    )
                    st.markdown(f"### {r.animal.name} the {r.animal.species} — {pct}% match  \n*{label}*")
                    st.caption(r.animal.info())
                st.progress(r.score)
                st.markdown("**Why:**")
                for f in r.top_factors:
                    sign = "✅" if f.contribution > 0 else "⚠️"
                    st.markdown(f"- {sign} {f.plain_english}")


def page_adopt():
    st.header("Adopt an Animal")
    shelter = st.session_state.shelter
    residents = shelter.residents
    if not residents:
        st.info("No animals available to adopt.")
        return

    names = [a.name for a in residents]
    chosen = st.selectbox("Animal to adopt", names)
    adopter_name = st.text_input("Adopter's name", value="")
    if st.button("Confirm adoption"):
        if not adopter_name.strip():
            st.error("Please enter the adopter's name.")
        else:
            msg = shelter.adopt(chosen, adopter_name.strip())
            st.success(msg)
            st.rerun()


def page_history():
    st.header("Adoption History")
    shelter = st.session_state.shelter
    all_animals = shelter.residents + shelter.adopted
    if not all_animals:
        st.info("No animals in the system yet.")
        return

    names = sorted({a.name for a in all_animals})
    chosen_name = st.selectbox("Look up an animal", names)
    animal = shelter.find(chosen_name)
    if animal is None:
        st.warning("Not found.")
        return

    st.subheader(f"{animal.name} the {animal.species}")
    if not animal.has_history():
        st.write("No adoption history yet.")
        return

    chain = animal.history.to_list()
    st.write(f"Recursive chain depth: **{len(chain)}**")
    for i, adopter in enumerate(chain):
        prefix = "Most recent" if i == 0 else f"{i} adoption(s) ago"
        st.markdown(f"**{prefix}:** {adopter}")


def page_about_model():
    st.header("About the Model")
    st.markdown(
        """
This shelter uses a **gradient boosting classifier** trained on 2,000
synthetic (animal, adopter) pairs. The training labels were generated
from a hidden ground-truth compatibility function encoding common-sense
shelter wisdom (energy matching, allergies, apartment-living constraints,
etc.). The model never sees the rules — it has to discover them from
labels.

**Why two models?** A logistic regression baseline scored 66% accuracy /
0.72 AUC. The gradient boosting model reached 82% / 0.82 — the gain
comes from non-linear interactions like *apartment AND large dog*, which
a linear model can't capture natively.

**Match explanations** are computed via local feature perturbation: for
each match, we flip one feature at a time and measure how the predicted
probability changes. The biggest swings become the "why" bullets in the
matcher.
        """
    )

    try:
        importances = pd.read_csv("models/feature_importances.csv")
        st.subheader("Top features the model learned to rely on")
        st.bar_chart(importances.head(10).set_index("feature"))
        with st.expander("Show full table"):
            st.dataframe(importances, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.info("Run `python train_model.py` to generate feature importances.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Shelter Match", page_icon="🐾", layout="wide")
    init_state()

    st.title("🐾 Animal Shelter Match")
    st.caption(
        "An ML extension of a C++ midterm project — recursive adoption "
        "history, polymorphic animal types, and a gradient boosting model "
        "that ranks animals by how well they fit each adopter's lifestyle."
    )

    page = st.sidebar.radio(
        "Navigate",
        ["Shelter", "Find a Match", "Adopt", "History", "About the Model"],
    )

    if page == "Shelter":
        page_shelter()
    elif page == "Find a Match":
        page_match()
    elif page == "Adopt":
        page_adopt()
    elif page == "History":
        page_history()
    else:
        page_about_model()


if __name__ == "__main__":
    main()
