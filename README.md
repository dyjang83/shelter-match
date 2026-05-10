# 🐾 Animal Shelter Match

An ML extension of a CS XL 32 C++ midterm project. The original assignment
asked for an animal shelter manager built around polymorphism, smart
pointers, and a recursive adoption-history chain. This version ports the
core OOP design to Python and adds a gradient boosting model that ranks
animals by how well they fit each adopter's lifestyle, served through a
Streamlit web UI — with a full evaluation pipeline including
cross-validation, calibration analysis, and a fairness audit.

**🔗 Live demo:** https://diane-shelter-match.streamlit.app 
**📓 Analysis notebook:** [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb)

## What's interesting here

- **Polymorphism doing real work.** In the original midterm, polymorphism
  only differentiated `printInfo()` output. Here, every `Animal` subclass
  contributes a feature vector that the matching model consumes — the
  abstract base class earns its keep beyond text formatting.
- **Hidden ground-truth function the model has to recover.** Training
  data is generated from a hand-designed compatibility function (energy
  matching, allergy hard-stops, apartment + large dog penalties, etc.).
  The model never sees these rules. Feature importances in the trained
  model recover them, which is shown in the analysis notebook.
- **Four-model baseline comparison with cross-validation.** Logistic
  regression, gradient boosting, XGBoost, and an MLP, all evaluated with
  5-fold stratified CV and reported with means + standard deviations.
- **Calibration analysis.** Reliability diagrams check whether predicted
  probabilities are meaningful (a "70% match" should mean 70% of those
  pairs are actually good matches), not just well-ranked.
- **Fairness audit.** Tests whether the model systematically
  disadvantages certain animal groups (rabbits, large dogs, low-trained
  animals) across a neutral pool of adopters, with disparity ratios for
  each protected group.
- **Per-match explanations via local perturbation.** For each ranked
  animal, we flip one feature at a time and measure the change in
  predicted probability — a lightweight SHAP-style attribution with
  zero extra dependencies.
- **Recursive adoption history preserved.** The C++ `unique_ptr`
  history chain ports cleanly to Python: each `AdoptionRecord` holds a
  reference to the previous record, and `print_history()` walks it
  recursively. The web UI renders the chain as a visual timeline.

## Headline results

5-fold cross-validation on 2,000 synthetic samples:

| Model               | Accuracy        | ROC-AUC         | F1              |
|---------------------|----------------:|----------------:|----------------:|
| Logistic Regression | 0.652 ± 0.015   | 0.693 ± 0.013   | 0.519 ± 0.018   |
| Gradient Boosting   | **0.816 ± 0.021** | **0.802 ± 0.017** | 0.596 ± 0.056 |
| XGBoost             | 0.817 ± 0.025   | 0.802 ± 0.012   | **0.621 ± 0.062** |
| MLP (32, 16)        | 0.757 ± 0.018   | 0.744 ± 0.020   | 0.556 ± 0.042   |

Gradient boosting is shipped in production. XGBoost is essentially tied
on accuracy but adds a heavy dependency for ~no real gain. The MLP
underperforms tree methods on this tabular dataset with only 2K samples
— a useful confirmation of the prior that gradient-boosted trees still
dominate small-tabular problems.

**Fairness findings:** rabbits face the largest disparity (4% strong-match
rate vs. ~9% for dogs/cats; disparity ratio 0.42), driven partly by genuine
ground-truth constraints and partly by underrepresentation in training
data. The notebook discusses what a production shelter would do about this.

## Project layout

```
shelter_ml/
├── shelter.py          # Animal/Dog/Cat/Rabbit/Adopter/Shelter (Python port of the C++ design)
├── data_gen.py         # Synthetic data generator + hidden ground-truth function
├── train_model.py      # Trains gradient boosting, saves model + feature importances
├── evaluation.py       # Cross-validation, calibration, per-species, failure modes
├── fairness.py         # Group disparity analysis with disparity-ratio metric
├── matcher.py          # Inference + per-match explanation via perturbation
├── app.py              # Streamlit web UI
├── notebooks/
│   └── analysis.ipynb  # Full analytical write-up with plots
├── data/
│   └── synthetic_matches.csv
├── models/
│   ├── match_model.joblib
│   └── feature_importances.csv
└── reports/
    ├── baseline_comparison.csv
    └── fairness_audit.csv
```

## Setup

```bash
pip install -r requirements.txt

# One-time: generate data, train model, run evaluation.
python data_gen.py
python train_model.py
python evaluation.py
python fairness.py

# Launch the web app.
streamlit run app.py

# Or open the analysis notebook.
jupyter notebook notebooks/analysis.ipynb
```

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

## What I'd do next

- Replace static label noise with realistic adoption-outcome data once available.
- Add Platt scaling or isotonic regression for calibration (`CalibratedClassifierCV`).
- Re-weight training data to upweight underrepresented species (rabbits).
- Add a "second-look" pass that surfaces the highest-scoring strong matches *per animal*, not just per adopter — so every animal gets visibility.
- Replace local perturbation explanations with proper SHAP values.
