# 🐾 Animal Shelter Match

**🔗 Live demo:** https://diane-shelter-match.streamlit.app

An ML extension of a CS XL 32 C++ midterm project. The original assignment
asked for an animal shelter manager built around polymorphism, smart
pointers, and a recursive adoption-history chain. This version ports the
core OOP design to Python and adds a gradient boosting model that ranks
animals by how well they fit each adopter's lifestyle, served through a
Streamlit web UI.

## What's interesting here

- **Polymorphism doing real work.** In the original midterm, polymorphism
  only differentiated `printInfo()` output. Here, every `Animal` subclass
  contributes a feature vector that the matching model consumes — so the
  abstract base class earns its keep beyond text formatting.
- **Hidden ground-truth function the model has to recover.** Training
  data is generated from a hand-designed compatibility function (energy
  matching, allergy hard-stops, apartment + large dog penalties, etc.).
  The model never sees these rules. Feature importances in the trained
  model recover them, which is shown in the "About the Model" page.
- **Linear vs. non-linear comparison.** Logistic regression hits 66%
  accuracy / 0.72 AUC; gradient boosting hits 82% / 0.82 by capturing
  interactions like *apartment AND large dog*.
- **Per-match explanations via local perturbation.** For each ranked
  animal, we flip one feature at a time and measure the change in
  predicted probability — a lightweight SHAP-style attribution with
  zero extra dependencies.
- **Recursive adoption history preserved.** The C++ `unique_ptr`
  history chain ports cleanly to Python: each `AdoptionRecord` holds a
  reference to the previous record, and `print_history()` walks it
  recursively. The web UI renders the chain as a visual timeline.

## Project layout

```
shelter_ml/
├── shelter.py        # Animal/Dog/Cat/Rabbit/Adopter/Shelter classes (Python port of the C++ design)
├── data_gen.py       # Synthetic data generator with hidden ground-truth function
├── train_model.py    # Trains logistic + gradient boosting; saves model + importances
├── matcher.py        # Inference + per-match explanation via perturbation
├── app.py            # Streamlit web UI
├── data/
│   └── synthetic_matches.csv
└── models/
    ├── match_model.joblib
    └── feature_importances.csv
```

## Setup

```bash
pip install -r requirements.txt

# Generate training data and train the model (one-time).
python data_gen.py
python train_model.py

# Launch the web app.
streamlit run app.py
```

Then open `http://localhost:8501`.

## Architecture notes

The shelter's OOP core is decoupled from the ML layer. `Animal.feature_dict()`
is the only seam — concrete subclasses contribute their own features, the
matcher consumes feature dicts without caring which concrete class
produced them. Adding a new species (say, `GuineaPig`) means subclassing
`Animal`, updating `SPECIES_TO_CLASS`, and regenerating training data.
The matcher and UI need no changes.

The Streamlit app keeps the `Shelter` instance and the loaded model in
`st.session_state` so they survive Streamlit's per-interaction script
reruns. Adoptions mutate the shelter in place; the recursive history
chain accumulates on the same Python object across multiple adoptions
of the same animal, matching the readmission behavior in the C++ sample
output.

## Model performance

| Model | Accuracy | ROC-AUC |
|-------|---------:|--------:|
| Logistic regression (baseline) | 0.66 | 0.72 |
| Gradient boosting (selected)   | **0.82** | **0.82** |

Top features the gradient boosting model relied on:
1. `energy_level` (animal)
2. `activity_level` (adopter)
3. `needs_quiet` (animal)
4. `hours_home_per_day` (adopter)
5. `prefers_quiet_home` (adopter)
6. `lives_in_apartment` (adopter)

These line up with the rules encoded in the hidden compatibility function,
which is the headline result.
