# Medical Assistant — Multi-Condition Hybrid ML/DL System

Generalizes the diabetes-only project into a config-driven medical
assistant. The same hybrid ML+DNN+SHAP+history+login pipeline now works
across multiple conditions — adding a new one means adding a dataset and
a config entry, not writing new routes or new training code.

## What's new in this version
- **Redesigned UI** — a consistent visual system (`static/style.css` +
  `templates/base.html`) across every page: deep-teal/amber-gold
  palette, Space Grotesk + IBM Plex Sans typography, card-based layout,
  color-coded risk results (green/red) with a probability bar and SHAP
  contribution list, and a heartbeat-line signature motif.
- **Fixed: login redirect bug** — visiting `/login` while already logged
  in used to re-render the login form (looked like login "wasn't
  working" when you clicked the nav link again after logging in). It
  now redirects straight to `/history`. Sessions are also now marked
  permanent (7-day cookie) so you don't get logged out mid-session.
- **New: Symptom Chat** (`/chatbot`) — a rule-based chatbot for general,
  common conditions (fever, cold, flu, headache, sore throat, upset
  stomach, food poisoning, allergies, cough, fatigue). Separate from the
  4 trained ML models — this doesn't use machine learning, it matches
  symptom keywords against a small knowledge base and always recommends
  professional care. See "About the Symptom Chat" below for exactly
  what it does and doesn't do.

## About the Symptom Chat — read this before relying on it
`chatbot.py` + `data/chatbot_kb.json` implement a **rule-based keyword
matcher**, not a language model and not a diagnostic tool:
- It checks your message against a list of ~20 emergency-symptom phrases
  first (chest pain, difficulty breathing, loss of consciousness,
  stroke signs, etc.) — if any match, it immediately tells you to seek
  emergency care and stops, without attempting home-care advice.
- Otherwise it matches your wording against keyword lists for 10 common,
  low-severity conditions and returns general self-care information plus
  "see a doctor if..." guidance for each.
- It never names or doses specific medications — only general categories
  ("over-the-counter pain relief", "ask a pharmacist").
- If nothing matches, it says so rather than guessing.

This is intentionally simple and inspectable — you can read the entire
knowledge base in `data/chatbot_kb.json` and see exactly what it will
say for any input. Extend it by adding entries to that file; no model
retraining involved. It is explicitly not a replacement for the 4
trained ML models above, and not a replacement for a doctor.

## Included conditions
| Condition | Dataset | Rows | Features |
|---|---|---|---|
| Diabetes | Pima Indians Diabetes | 768 | 8 |
| Heart Disease | UCI Cleveland Heart Disease | 303 | 13 |
| Breast Cancer | Wisconsin Breast Cancer (sklearn built-in) | 569 | 30 |
| Liver Disease | Indian Liver Patient Dataset (ILPD) | 579 | 10 |

Adding **Liver Disease** to this project took exactly two steps and zero
code changes — proof the architecture generalizes:
1. Dropped `data/liver_disease.csv` in (10 numeric features, binary
   `liver_disease` target)
2. Added one entry to `config/diseases.json`, then ran
   `python train.py liver_disease`

Everything else — the form, the `/predict` route, SHAP explanations,
history logging — worked immediately, because none of it references a
specific condition.

Add a fifth the same way: any binary-outcome, numeric-feature CSV works
out of the box (kidney disease, stroke risk, Parkinson's, etc.). What
this architecture does *not* handle without real changes: multi-class
diagnosis, image input (X-rays, scans — needs a CNN), or free-text
symptoms (needs NLP) — see the last section.

## Folder structure
```
med-assistant/
├── config/diseases.json      # condition registry — the single source of truth
├── config_loader.py          # loads/validates the registry
├── app.py                    # Flask app: dynamic routes per condition + chatbot
├── train.py                  # generic trainer: python train.py [condition ...]
├── utils.py                  # generic hybrid prediction + SHAP, parametrized by condition
├── chatbot.py                 # rule-based general symptom checker
├── dl_model.py                # shared DNN architecture
├── database.py                # SQLite history, tagged by condition
├── streamlit_app.py           # alternative UI with a condition dropdown
├── requirements.txt
├── static/style.css           # shared design system
├── data/
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── breast_cancer.csv
│   ├── liver_disease.csv
│   └── chatbot_kb.json        # symptom chat knowledge base
├── models/<condition>/         # best_ml_model.pkl, dnn_model.keras, scaler.pkl,
│                                 # feature_columns.pkl, shap_background.pkl, meta.json
├── reports/<condition>/        # cv_comparison.*, test_comparison.*, roc_curves.png, confusion_matrix.png
└── templates/
    ├── base.html                 # shared nav, fonts, heartbeat signature
    ├── home.html                 # lists all conditions
    ├── form.html                  # form auto-generated from the condition's feature list
    ├── login.html
    ├── history.html                # filterable by condition
    └── chatbot.html                # symptom chat UI
```

## What generalized vs. what's condition-specific
- **Generalized (shared code, zero hardcoding):** training pipeline (CV
  model selection, DNN training, performance-weighted hybrid, SHAP
  explanation, plots), Flask routes, the prediction form (fields are
  read from `feature_columns.pkl`, not hand-written HTML), history
  storage/filtering, the Streamlit UI.
- **Condition-specific (lives only in `config/diseases.json` + `data/`):**
  the dataset file, the target column name, and the two display labels
  (e.g. "High Diabetes Risk" vs "Malignant").

## Setup
```bash
pip install -r requirements.txt
```

## 1. Train
```bash
python train.py                       # trains all three conditions
python train.py diabetes               # or just one
python train.py diabetes heart_disease # or a subset
```
For each condition this runs 5-fold CV to pick the best ML model, trains
a DNN, computes performance-weighted hybrid weights from a validation
split, and evaluates all three (ML, DNN, hybrid) on a held-out test set.
Saves everything under `models/<condition>/` and `reports/<condition>/`.

## 2. Run the web app
```bash
python app.py
```
`/` lists all conditions. Each links to `/<condition>`, a form built
from that condition's actual feature list — the diabetes form has 8
number fields, the breast cancer form has 30, generated from the same
template. `/chatbot` opens the general Symptom Chat. `/history` (behind
`/login`, demo creds `admin`/`admin123`, override via
`ADMIN_USERNAME`/`ADMIN_PASSWORD`) shows predictions across all
conditions, filterable by condition.

## 3. Or run the Streamlit UI
```bash
streamlit run streamlit_app.py
```
Pick a condition from the dropdown; the form and prediction logic adapt
automatically.

## Honest results — the hybrid doesn't uniformly win here
On this run's held-out test sets:

| Condition | Best ML (F1) | DNN (F1) | Weighted Hybrid (F1) |
|---|---|---|---|
| Diabetes | 0.655 (LogReg) | 0.603 | 0.632 |
| Heart Disease | 0.833 (RandomForest) | 0.806 | 0.794 |
| Breast Cancer | 0.964 (LogReg) | 0.951 | 0.963 |
| Liver Disease | 0.825 (RandomForest) | 0.761 | 0.800 |

The hybrid doesn't uniformly beat the best single model — same honest
finding as before. It comes closest to matching or beating both models
on Diabetes and Breast Cancer; on Heart Disease and Liver Disease it
lands between the two rather than above them. Two reasons this happens:
1. **The datasets are small** (303–768 rows). Validation-based weights
   computed on ~100–150 rows carry real sampling noise, and the
   resulting weight (usually landing close to 50/50) doesn't always
   match which model actually generalizes better to the untouched test
   set.
2. **Which ML model wins CV selection now varies by condition** (Random
   Forest for Heart Disease and Liver Disease, Logistic Regression for
   Diabetes and Breast Cancer) — a healthy sign that model selection is
   actually responding to each dataset rather than always picking the
   same algorithm.

## Fixed: class imbalance on Liver Disease (SMOTE)

Liver Disease is imbalanced (414 disease-positive vs 165 negative
records). The first version of this pipeline had all three models
predicting "disease" too eagerly — high recall on the positive class
(~94–96%) but very poor recall on the negative class (~18–27%), meaning
it was much better at catching disease than confirming a healthy
patient. F1 alone hid this; only the confusion matrix showed it.

**Fix:** every model pipeline now oversamples the minority class with
SMOTE (`imblearn.pipeline.Pipeline` + `SMOTE`), applied only to the data
each model actually trains on — never to the validation set, the test
set, or a cross-validation fold's held-out portion, so there's no data
leakage. This is applied generically to all four conditions, not
special-cased for liver disease, since class imbalance is a property any
dataset can have.

**Before → after on Liver Disease (hybrid model, held-out test set):**
| Metric | Before SMOTE | After SMOTE |
|---|---|---|
| Recall — disease-negative class | 0.21 | 0.52 |
| F1 — disease-negative class | 0.33 | 0.51 |
| F1 — disease-positive class (overall) | 0.847 | 0.800 |

The negative-class recall more than doubled — the model is now
meaningfully better at correctly identifying healthy patients, which is
the failure mode that actually matters for a diagnostic tool (missing
disease is worse than a false alarm, but a model that never says
"healthy" isn't useful either). The tradeoff is a small drop in the
overall F1 number, because it's no longer getting an inflated score by
defaulting to "disease" — that's the expected, correct tradeoff, not a
regression. See `reports/liver_disease/confusion_matrix_hybrid.png`
for the full picture.

If you need the hybrid to clearly beat the ML model on every condition,
the honest next step is either more data, or restricting the hybrid to
only fire when it beats the ML-only model on the validation split
(falling back to ML-only otherwise) — I can add that if you want it.

## Still-open ideas
- Per-condition hyperparameter tuning (GridSearchCV)
- A "use hybrid only if it validates better than ML-only" fallback rule
- Proper multi-user accounts instead of one shared admin login
- Deployment config (Docker, gunicorn)
- Decision-threshold tuning (currently fixed at 0.5) as an alternative/complement to SMOTE
