# Explainable Ensemble Machine Learning for Early Chronic Kidney Disease Prediction

**ML for Social Good Ensemble Challenge — Mission Health**
CIA-3, Machine Learning (MCA 521-4)

> This model is a screening support tool only, not a medical diagnostic system.

## 1. Problem and Impact Framing

**Problem:** Chronic Kidney Disease (CKD) is often symptomless until it is advanced. This project builds a binary
classifier (`CKD` vs `Not CKD`) to support earlier screening in under-resourced clinics.

**Beneficiaries:** patients, rural/under-resourced clinics, primary healthcare workers, public health bodies.

**Prediction target:** binary classification — `CKD` vs `Not CKD`.

**Measurable impact:** more patients flagged for a confirmatory renal lab panel *before* symptoms appear, using
only cheap, widely available measurements — no lab test required to get a risk flag.

**Why ML is suitable:** CKD risk depends on non-linear interactions between comorbidities (e.g. hypertension *and*
diabetes together carrying more risk than either alone) — a pattern well suited to ML, especially ensembles.

**Unit of analysis:** one patient record (one row = one patient's clinical results at a single visit).

**Responsible-use limitations:** small, single-hospital dataset; not externally validated; must never replace a
clinician's diagnosis — output is only a screening flag for further testing.

## 2. Dataset

- **Source:** [UCI Chronic Kidney Disease Data Set](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
  (id=336), fetched via the `ucimlrepo` package and cached in `data/chronic_kidney_disease_raw.csv`.
- 400 patients, 24 raw features, missing values present in nearly every column, target split ~62% CKD / 38% Not
  CKD (moderate imbalance).
- Collected at a hospital in Tamil Nadu, India — a single site, over a short window (see Limitations).

### Key decision: why only 6 of the 24 features are used for modelling

Most of the remaining 18 features (serum creatinine, specific gravity, albumin, hemoglobin, urine microscopy, ...)
are the clinical lab panel used to **diagnose** CKD, not independent risk factors — serum creatinine directly
determines the diagnostic eGFR calculation, and albuminuria is itself a diagnostic criterion. Training on them
makes the model reproduce the diagnosis rather than predict *early* risk. The notebook proves this empirically
(Section 2.5): a plain, untuned Logistic Regression trained on the full 24-feature set already reaches
**ROC-AUC = 1.0000** under 5-fold CV — confirming this is a property of the features (they leak the diagnosis),
not a coincidence of one lucky train/test split. We therefore model only `age`, `bp`, `bgr`, `htn`, `dm`, `cad` —
everything a resource-limited clinic can collect **without** ordering a renal lab test — plus one engineered
feature. This also makes the baseline-vs-ensemble comparison meaningful again: on the full feature set every
model (including the baseline) saturates near 100% and there is no real gap left for the ensembles to demonstrate.

## 3. Data Wrangling — Audit and Decisions

Each cleaning step is a deliberate, justified choice, not just default boilerplate:

| Issue | Finding | Decision |
|---|---|---|
| Missing values | Present in nearly every column | Median imputation (numeric) / most-frequent (categorical), fit only on training data inside the pipeline |
| Duplicate rows | 0 found (checked *after* text cleaning, so a `'ckd'` vs `'ckd\t'` near-duplicate pair would still be caught) | Drop if found |
| Invalid text | `'ckd\t'`, `'\tno'`, `'\tyes'` (stray tabs from hand-entered records) | Stripped/normalised before anything else |
| Invalid numeric values | 0 found for `age` (0-110), `bp` (≥0), `bgr` (≥0) | Guard kept anyway so the same code stays correct on a future data batch |
| Outliers | IQR flags 10 (age), 36 (bp), 34 (bgr) values | **Kept, not removed** — an extreme BP/glucose reading is itself a genuine risk signal in exactly the at-risk patients this model needs to catch. Tree models (RF/XGBoost) are naturally robust to this skew; `StandardScaler` keeps the LR baseline stable |
| Class imbalance | ~62% CKD / 38% Not CKD | SMOTE, applied **only inside the training fold** of every model pipeline — never to validation/test |

## 4. EDA — Key Observations

- Blood pressure and blood glucose are visibly higher in CKD patients; hypertension/diabetes `yes` responses are
  concentrated in the CKD group — consistent with these being the two leading real-world causes of CKD.
- These differences are visibly *less* clean-cut than the full diagnostic lab panel would show — which is the
  point: it makes this a genuine risk-prediction problem instead of a disguised re-diagnosis.

## 5. Feature Engineering

One engineered feature, from a fixed clinical rule (not a statistic learned from the data, so it cannot leak
information across the train/test split):

- **`htn_dm_comorbidity`** = 1 if a patient has both hypertension **and** diabetes, else 0. These are the two
  leading causes of CKD, and their *co-occurrence* carries materially higher risk than either alone — a signal a
  linear model cannot extract from the two separate `htn`/`dm` columns without an explicit interaction term.

## 6. Train/Validation/Test Split and Preprocessing Pipeline

- **Split:** 70% train / 15% validation / 15% test (280 / 60 / 60 patients), **stratified** so the ~62/38 class
  ratio is preserved in every split.
- **Leakage-safety rule:** the test set is touched **exactly once**, at final evaluation (Section 3, step 6).
  Validation is used only as a sanity check while building/tuning each model.
- **Pipeline:** one scikit-learn/imblearn `Pipeline` per model — `ColumnTransformer` (median imputation + scaling
  for numeric, most-frequent imputation + one-hot encoding for categorical) → SMOTE → classifier. Fit only on
  `X_train`, so every statistic (medians, means/std for scaling, SMOTE neighbours) comes from training data alone.

## 7. Models, Architecture, and Tuning

| Role | Model |
|---|---|
| Baseline | Logistic Regression (untuned, per the assignment brief) |
| Bagging | Random Forest |
| Boosting | XGBoost |
| Heterogeneous ensemble | Stacking: LR + Random Forest + XGBoost base learners → Logistic Regression meta-model |

**Tuning:** `RandomizedSearchCV` (5-fold stratified CV, scored on ROC-AUC, fit on training data only) over:
- Random Forest — `n_estimators`, `max_depth`, `min_samples_split`. Best found: `max_depth=None, min_samples_split=8, n_estimators=221`.
- XGBoost — `learning_rate`, `max_depth`, `n_estimators`. Best found: `learning_rate≈0.0102, max_depth=6, n_estimators=376`.

**Two deliberate design decisions in the Stacking model** (this is where the tightest leakage-safety requirement
in the brief applies — "prevent leakage in meta-learning"):
1. **Reuses the tuned RF/XGBoost hyperparameters** above, instead of fresh defaults, so the ensemble actually
   benefits from the tuning step rather than being built from untuned base learners.
2. **SMOTE is nested inside each base learner's own sub-pipeline**, not applied once before stacking. If SMOTE ran
   once beforehand, a synthetic point and the real points it was interpolated from could land on opposite sides of
   `StackingClassifier`'s internal 5-fold CV split — a subtle leak between the folds used to train the meta-model.
   Resampling separately inside each internal fold avoids that.

## 8. Results (held-out test set, touched once)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (Logistic Regression) | 0.783 | 1.000 | 0.649 | 0.787 | 0.918 |
| Bagging (Random Forest, tuned) | 0.933 | 0.971 | 0.919 | 0.944 | 0.984 |
| Boosting (XGBoost, tuned) | 0.917 | 0.971 | 0.892 | 0.930 | **0.985** |
| Stacking (LR+RF+XGB, tuned base learners) | 0.933 | 0.971 | 0.919 | 0.944 | 0.984 |

**Ensembles clearly outperform the baseline**, mainly by fixing its weak Recall (0.649 → 0.89-0.92) — the baseline
misses roughly a third of true CKD cases, which is the worst failure mode for a screening tool; every ensemble
closes most of that gap while keeping Precision high. XGBoost is deployed (best ROC-AUC, 0.985, +0.068 over
baseline), though the top three ensembles are close enough on a 60-patient test set that this should be read as
"all three architectures clearly beat the baseline," not "XGBoost is definitively the best architecture."

**On the baseline's perfect 1.000 Precision** (this is *not* a leakage red flag): the test-set confusion matrix is
TN=23, FP=0, FN=13, TP=24. It means the baseline never raises a false alarm on the 24 patients it's confident
enough to flag, but stays silent on 13 of the 37 real CKD cases — exactly the Recall weakness above, and a small,
statistically unremarkable result for a linear model on 6 features (unlike the earlier full-feature version of
this project, where *every* model hit ~100% on *every* metric, which *was* a red flag and is why the feature set
was restricted — see Section 2).

## 9. Explainability (SHAP)

Explains the tuned XGBoost model specifically (`TreeExplainer` gives fast, exact explanations for tree ensembles).

- **Global:** blood pressure, glucose, and the hypertension/diabetes comorbidity flag are the strongest drivers of
  predicted CKD risk — consistent with hypertension and diabetes being the two leading real-world causes of CKD.
- **Local (synthetic patient):** age 55, BP 90 mm/Hg, glucose 180 mg/dl, hypertensive, diabetic, no CAD →
  **predicted CKD, 98.2% probability.** The SHAP waterfall plot shows this driven mainly by the elevated BP/glucose
  and the hypertension+diabetes comorbidity — giving a clinician a transparent reason to order a confirmatory
  renal panel, not just a black-box score.

## 10. Ethics and Limitations

- **Bias:** single hospital, single region, short collection window — may not generalise elsewhere.
- **Fairness:** sex, ethnicity and socioeconomic status are not in the data, so subgroup fairness could not be
  checked here and should be validated before real deployment.
- **Privacy:** data is de-identified, but age + comorbidity history could still be re-identifying in a small
  clinic; real deployment must follow health-data privacy regulation.
- **False negatives** (a missed CKD case) are more costly than false positives, which is why Recall/ROC-AUC, not
  just Accuracy, guided model selection throughout.
- **Human oversight:** the model must never auto-generate a diagnosis; every flagged case should go to a clinician
  for a confirmatory renal panel.
- **Uncertainty:** with only 400 training patients and 6 input features, predicted probabilities are approximate,
  especially for patient profiles rare in the training data.
- **Deployment limits:** not externally validated, not a regulatory-approved medical device — a research/screening
  prototype only.

> **This model is intended only as a screening support tool and not as a medical diagnostic system.**

## 11. Files

- `CIA3.ipynb` — the full notebook, structured around the 5 assignment questions (Q1–Q5); re-executes top to
  bottom with zero errors.
- `data/chronic_kidney_disease_raw.csv` — cached raw dataset (regenerated automatically if missing).
- `models/ckd_best_pipeline.joblib` — the trained, deployable pipeline (preprocessing + SMOTE + model), produced
  by Section 3 of the notebook.
- `requirements.txt`

## 12. Installation and Execution

```bash
pip install -r requirements.txt
```

1. Open `CIA3.ipynb` in Jupyter/VS Code and run all cells top to bottom, or:
   ```bash
   python -m nbconvert --to notebook --execute --inplace CIA3.ipynb
   ```
2. This regenerates `data/` (if missing) and `models/ckd_best_pipeline.joblib`. All randomness is seeded
   (`RANDOM_STATE = 42`) for reproducibility.


