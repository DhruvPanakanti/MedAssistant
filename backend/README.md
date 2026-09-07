# Medical Assistant — Multi-Condition Hybrid ML/DL System

A config-driven medical risk-assessment platform. Each supported
condition is predicted by a hybrid model (a tuned classical ML
classifier + a deep neural network, combined by validation-weighted
averaging) with SHAP-based explanations, served through a Flask web
app, a Streamlit alternative UI, and a JSON API. Adding a new condition
requires a dataset and a config entry — no new routes, templates, or
training code.

## Included conditions
| Condition | Dataset | Records | Features |
|---|---|---|---|
| Diabetes | Pima Indians Diabetes | 768 | 8 |
| Heart Disease | UCI Cleveland Heart Disease | 303 | 13 |
| Breast Cancer | Wisconsin Breast Cancer (sklearn built-in) | 569 | 30 |
| Liver Disease | Indian Liver Patient Dataset (ILPD) | 579 | 10 |

Full dataset sourcing (original publisher, record counts, links) is
available at `/citations` (linked from the footer on every page).

## Features
- **Hybrid prediction** — for each condition, `train.py` grid-searches
  hyperparameters (5-fold CV) across Logistic Regression, Decision Tree,
  Random Forest, and Gradient Boosting, trains a companion DNN, and
  combines both into a validation-weighted hybrid probability.
- **SHAP explanations, in plain language** — every prediction can return
  the top factors that drove it as readable sentences (e.g. "Chest Pain
  Type (Asymptomatic) strongly decreased the risk"), not raw feature
  names or impact numbers.
- **General lifestyle tips** — each condition's page shows general,
  non-personalized dietary and lifestyle guidance (foods to favor/limit,
  other lifestyle factors), also included in the PDF report. This is
  static educational content, not derived from the model or a specific
  prediction, and is clearly labeled as not a substitute for advice from
  a doctor or dietitian.
- **Class-imbalance handling** — every model pipeline oversamples the
  minority class with SMOTE during training only, via
  `imblearn.pipeline.Pipeline`, so validation and test data are never
  touched by synthetic samples.
- **Adjustable decision threshold** — each condition has a
  `recommended_threshold` (Youden's J statistic on validation data) and
  an admin-tunable `applied_threshold` (default 0.5), changeable at
  `/admin/thresholds` without retraining. The page explains what a
  threshold means and what the current value does, in plain language.
- **Data-confidence indicator** — predictions report `limited` /
  `moderate` / `adequate` confidence based on how many records the
  condition was actually trained on.
- **Input validation** — every field's valid range or categorical
  options are derived directly from the training data. Categorical
  fields accept plain text (e.g. "Male", "Yes") with autocomplete
  suggestions, converted server-side to the model's numeric encoding.
- **PDF report** — downloadable one-page summary of any prediction
  (inputs, result, confidence, contributing factors, lifestyle tips).
- **CSV batch prediction** — upload many patients at once at
  `/<condition>/batch`, get a results CSV back (500-row limit, template
  download provided).
- **Prediction history** — every prediction is logged to SQLite,
  browsable by an admin at `/history`, filterable by condition.
- **Symptom Chat** — a separate, rule-based chatbot (`/chatbot`) for
  general, non-diagnostic guidance on common conditions like fever,
  cold, or headache. See "About the Symptom Chat" below.
- **Accessible, responsive UI** — skip-to-content link, semantic
  landmarks, `aria-live` result announcements, WCAG AA-checked color
  contrast, keyboard-navigable throughout.
- **Automated tests** — a pytest suite (57 tests) covering routes,
  prediction correctness, validation, authentication, the chatbot,
  threshold tuning, batch upload, PDF generation, lifestyle tips, and
  the Streamlit UI (via Streamlit's own `AppTest` framework — an actual
  end-to-end run of the app, not just a syntax check).

## About the Symptom Chat
The Symptom Chat combines a hardcoded safety layer with an optional
LLM-backed conversational layer, in that order:

1. **Emergency detection (`chatbot.py`) is always hardcoded and always
   runs first.** It checks the message against a list of
   emergency-symptom phrases (chest pain, difficulty breathing, loss of
   consciousness, stroke signs, etc.). If any match, it immediately
   recommends emergency care and stops — this check never depends on,
   or is reachable by, the LLM layer below, so it can't be talked around
   by a prompt.
2. **Optional LLM layer (`gemini_chat.py`), using Google's Gemini API.**
   If a `GEMINI_API_KEY` is configured, general (non-emergency)
   conversation is handled by Gemini, constrained by a strict system
   prompt to health/symptom topics only, with a code-enforced check (not
   just prompt instructions) that catches off-topic requests and
   replaces them with our own consistent redirect message rather than
   trusting the model's own wording. It never names specific medications
   or dosages, never diagnoses, and always recommends professional care
   for anything serious. Gemini's internal "thinking" is explicitly
   constrained (not left at its slow, verbose default) — but the exact
   parameter for this differs by model generation and the two are
   mutually exclusive: Gemini 2.x uses `thinking_budget` (a token
   count, set to 0), Gemini 3.x replaced that with `thinking_level` (a
   string level, set to `"low"`) and rejects `thinking_budget` outright
   with a hard error. `gemini_chat.py` detects the configured model's
   generation from its name and sends the correct parameter
   automatically, so switching `GEMINI_MODEL` between generations (as
   this project has already needed to do once) doesn't break the
   chatbot.
3. **Rule-based fallback (`data/chatbot_kb.json`).** If no API key is
   configured, or the Gemini call fails for any reason, the chatbot
   silently falls back to matching the message against a small,
   readable knowledge base of common conditions — the app works fully
   without any API key at all.

### Setting up the LLM layer (optional)
Open the `.env` file in the project root, uncomment the `GEMINI_API_KEY`
line, and paste in your key:
```
GEMINI_API_KEY=your-key-here
```
That's it — `config_loader.py` loads `.env` automatically every time
the app starts, so you only need to do this once. No need to set an
environment variable manually each session.

Get a free key at https://aistudio.google.com/apikey. **Never commit
your `.env` file or paste your key into chat** — `.env` is already
excluded by `.gitignore`, and `.env.example` (safe, no real key) is
the version that gets committed instead. On Google's free tier,
prompts and responses may be used to improve their products; a paid
tier removes this. Optionally uncomment `GEMINI_MODEL` to override the
default model (`gemini-3.6-flash`).

The same `.env` file also holds optional overrides for
`ADMIN_USERNAME`, `ADMIN_PASSWORD`, and `FLASK_SECRET_KEY` — all
commented out by default so the app's built-in defaults keep working
until you deliberately uncomment and change them.

### If the chatbot seems to only give rule-based answers
The Symptom Chat is one chatbot with three layers, tried in order:
hardcoded emergency detection (always first) → Gemini, if configured
and working → the rule-based knowledge base as a fallback. If Gemini
never connects, every message silently falls through to the rule-based
layer — which is safe (the chatbot never breaks), but gives no
indication of why. Run this to find out exactly what's wrong:
```bash
python check_gemini.py
```
It checks, in order, whether `.env` exists, whether `GEMINI_API_KEY` is
set (and isn't still the placeholder), whether `google-genai` is
installed, and finally makes a real test call to Gemini — reporting the
exact point of failure instead of a raw traceback. When the app itself
is running (`python app.py`), the same underlying failures are also
logged to the console.

The chatbot is not a replacement for the trained risk models above, and
not a replacement for a doctor.

## Folder structure
```
med-assistant/
├── config/diseases.json      # condition registry — the single source of truth
├── config_loader.py          # loads/validates the registry, defines PROJECT_ROOT
├── app.py                    # Flask app: routes per condition + chatbot + admin + batch
├── train.py                  # generic trainer: python train.py [condition ...]
├── utils.py                  # hybrid prediction + SHAP + validation + threshold logic
├── pdf_report.py              # per-prediction PDF report generator (reportlab)
├── lifestyle_tips.py           # loader for general dietary/lifestyle guidance
├── chatbot.py                 # symptom checker: hardcoded emergency detection + rule-based fallback
├── gemini_chat.py              # optional LLM-backed conversational layer (Gemini)
├── check_gemini.py              # standalone diagnostic: python check_gemini.py
├── dl_model.py                # shared DNN architecture
├── database.py                # SQLite history, tagged by condition
├── streamlit_app.py           # alternative UI with a condition dropdown
├── requirements.txt
├── .env                        # your local secrets (gitignored) — copy from .env.example
├── .env.example                 # safe template committed to git, no real values
├── static/style.css           # shared design system
├── data/
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── breast_cancer.csv
│   ├── liver_disease.csv
│   ├── chatbot_kb.json        # symptom chat knowledge base
│   ├── citations.json          # dataset sourcing info for /citations
│   └── lifestyle_tips.json      # general dietary/lifestyle guidance per condition
├── models/<condition>/         # best_ml_model.pkl, dnn_model.keras, scaler.pkl,
│                                 # feature_columns.pkl, shap_background.pkl, meta.json
├── reports/<condition>/        # cv_comparison.*, test_comparison.*, roc_curves.png, confusion_matrix.png
├── tests/                       # pytest suite
└── templates/
    ├── base.html                 # shared nav, footer, fonts, heartbeat signature, skip link
    ├── home.html                 # lists all conditions
    ├── form.html                  # form auto-generated from the condition's feature list
    ├── login.html
    ├── history.html                # filterable by condition
    ├── chatbot.html                # symptom chat UI
    ├── citations.html               # dataset sources (linked from the footer)
    ├── batch.html                   # CSV batch upload
    └── admin_thresholds.html        # admin decision-threshold tuning
```

## What generalized vs. what's condition-specific
- **Generalized (shared code, zero hardcoding):** training pipeline (CV
  model selection, hyperparameter tuning, DNN training, SMOTE, hybrid
  weighting, SHAP explanation, plots), Flask routes, the prediction form
  (fields are read from the trained model's feature list, not
  hand-written HTML), input validation, history storage/filtering, PDF
  generation, batch upload, the Streamlit UI.
- **Condition-specific (lives only in `config/diseases.json` +
  `data/`):** the dataset file, the target column name, the two outcome
  display labels, and the lifestyle tips content.

## Setup
```bash
pip install -r requirements.txt
```
`requirements.txt` pins exact versions matching what trained the model
files included in `models/`. If you already have a virtual environment
with different package versions installed, you may see
`InconsistentVersionWarning` messages when loading the shipped `.pkl`
files — harmless, but the reliable fix is to retrain locally against
your installed versions:
```bash
python train.py
```

## 1. Train
```bash
python train.py                        # trains all conditions
python train.py diabetes                # or just one
python train.py diabetes heart_disease  # or a subset
```
For each condition this grid-searches ML hyperparameters via 5-fold CV,
trains a DNN, computes validation-weighted hybrid weights and a
recommended decision threshold, and evaluates ML / DNN / hybrid on a
held-out test set. Saves everything under `models/<condition>/` and
`reports/<condition>/`.

## 2. Run the web app
```bash
python app.py
```
`/` lists all conditions. Each links to `/<condition>`, a form built
from that condition's actual feature list, followed by general
lifestyle tips. `/chatbot` opens the Symptom Chat. `/<condition>/batch`
opens CSV batch upload. `/citations` (footer link) lists dataset
sources. `/history` and `/admin/thresholds` require login (`/login`,
demo credentials `admin`/`admin123`, override via
`ADMIN_USERNAME`/`ADMIN_PASSWORD` environment variables).

## 3. Or run the Streamlit UI
```bash
streamlit run streamlit_app.py
```

## Running the tests
```bash
pytest              # run everything
pytest -v            # verbose
pytest tests/test_prediction.py   # a single file
```
Tests assume trained models already exist in `models/` (run `train.py`
first) and run against them via Flask's test client — no live server
needed.

## Model evaluation (held-out test set)
| Condition | Best ML model (F1) | DNN (F1) | Weighted Hybrid (F1) |
|---|---|---|---|
| Diabetes | 0.672 (Random Forest) | 0.602 | 0.643 |
| Heart Disease | 0.815 (Random Forest) | 0.754 | 0.807 |
| Breast Cancer | 0.964 (Logistic Regression) | 0.976 | 0.976 |
| Liver Disease | 0.820 (Gradient Boosting) | 0.730 | 0.772 |

Which ML algorithm wins varies by condition, reflecting per-dataset
model selection rather than a fixed default. The hybrid ensemble does
not uniformly outperform the best single model — it lands between the
ML and DNN components (below the best single model) on Diabetes and
Liver Disease, close behind on Heart Disease, and ties the DNN just
above the ML model on Breast Cancer. Because DNN training has some
inherent run-to-run variance even with fixed random seeds, exact
numbers will shift slightly if you retrain — rerun `train.py` and check
`reports/<condition>/test_comparison.csv` for your own numbers rather
than treating the table above as fixed. This reflects the modest size
of these datasets (303–768 records): validation-based hybrid weights
computed on roughly 100–150 rows carry real sampling noise. Full
per-condition metrics, ROC curves, and confusion matrices are in
`reports/<condition>/`.

Liver Disease's training data is imbalanced (~2.5:1 positive-to-negative
ratio); SMOTE oversampling during training improves negative-class
recall relative to a naive baseline, at a modest, expected cost to the
aggregate F1 score — see `reports/liver_disease/confusion_matrix_hybrid.png`.

## Known limitations
- **Heart Disease's `thal` field shows raw numeric codes (0–3), not
  readable labels**, unlike every other categorical field. This is
  deliberate, not an oversight: the source dataset
  (`kb22/Heart-Disease-Prediction` on GitHub) never documents what these
  codes mean, and published mappings for similarly-named heart disease
  datasets disagree with each other and don't account for all 4 values
  present in this file. A correlation check against the training data
  (value 2 has the lowest heart-disease rate, consistent with "Normal")
  supports a partial guess, but not a confident label for values 1 vs.
  3. Showing an incorrect clinical label would be worse than showing a
  plain number, so it stays unlabeled — see the comment above
  `CATEGORY_LABELS["heart_disease"]` in `utils.py` — until the source
  encoding can be confirmed authoritatively.

## Roadmap
- A "use hybrid only if it validates better than ML-only" fallback rule
- Proper multi-user accounts instead of one shared admin login
- Deployment configuration (Docker, gunicorn)
- LLM-powered conversational symptom chat (the current rule-based
  chatbot trades conversational flexibility for guaranteed, inspectable
  output)
