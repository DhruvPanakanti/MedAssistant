"""
Loads trained artifacts for any configured condition and produces hybrid
(ML + DL) predictions with an optional SHAP-based explanation.
"""
import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

from config_loader import get_disease, PROJECT_ROOT

_cache = {}
_spec_cache = {}

# Human-readable labels for categorical codes, sourced from the documented
# UCI Cleveland Heart Disease and Indian Liver Patient (ILPD) attribute
# definitions, and cross-checked against this project's actual data
# (e.g. liver Gender 1/0 split matches ILPD's well-known ~439/142 male/female
# split, confirming direction). `thal` is deliberately left unlabeled beyond
# its raw code: this specific heart.csv has a known data-entry artifact
# (thal=0 appears in only 2 of 303 rows and isn't a real category per the
# source documentation, and different published copies of this file disagree
# on which remaining code means "normal" vs "defect"), so asserting a
# specific clinical meaning for every value risks being confidently wrong,
# the same way the flipped `target` column was.
CATEGORY_LABELS = {
    "heart_disease": {
        "sex": {0: "Female", 1: "Male"},
        "cp": {
            0: "Typical angina", 1: "Atypical angina",
            2: "Non-anginal pain", 3: "Asymptomatic",
        },
        "fbs": {0: "No", 1: "Yes"},
        "restecg": {
            0: "Normal", 1: "ST-T wave abnormality",
            2: "Left ventricular hypertrophy",
        },
        "exang": {0: "No", 1: "Yes"},
        "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
        "ca": {
            0: "0 vessels", 1: "1 vessel", 2: "2 vessels",
            3: "3 vessels", 4: "4 vessels (rare in training data)",
        },
        # "thal" is deliberately NOT labeled here. The source dataset
        # (kb22/Heart-Disease-Prediction on GitHub) uses raw codes 0-3
        # with no documented meaning anywhere in that repo. Published
        # mappings for similarly-named heart disease datasets disagree
        # with each other (some say 0=Normal/1=Fixed/2=Reversible, only
        # covering 3 values; others say 1=Fixed/2=Normal/3=Reversible;
        # the original UCI Statlog docs use an unrelated 3/6/7 scale) —
        # and none of them account for all 4 values actually present in
        # this file. A correlation check against our own data (value 2
        # has by far the lowest heart-disease rate, consistent with
        # "Normal") supports a partial guess, but not a confident
        # assignment of "Fixed" vs "Reversible" between values 1 and 3.
        # Showing a wrong clinical label would be worse than showing a
        # plain number, so this stays unlabeled until the source
        # encoding can be confirmed authoritatively.
    },
    "liver_disease": {
        "Gender": {0: "Female", 1: "Male"},
    },
}

# Human-readable field labels, shown in place of raw column names on the
# form. Not every feature needs an entry — anything missing falls back to
# `_humanize()`, which turns "mean radius" / "Total_Bilirubin" into
# "Mean Radius" / "Total Bilirubin". Cryptic clinical abbreviations
# (cp, fbs, trestbps, thalach...) DO need an explicit entry since no
# generic formatting makes those readable.
FEATURE_LABELS = {
    "heart_disease": {
        "cp": "Chest Pain Type",
        "trestbps": "Resting Blood Pressure (mm Hg)",
        "chol": "Cholesterol (mg/dl)",
        "fbs": "Fasting Blood Sugar > 120 mg/dl",
        "restecg": "Resting ECG Result",
        "thalach": "Max Heart Rate Achieved",
        "exang": "Exercise-Induced Angina",
        "oldpeak": "ST Depression (Exercise vs Rest)",
        "slope": "ST Segment Slope",
        "ca": "Major Vessels Colored by Fluoroscopy",
        "thal": "Thalassemia Test Result (raw lab code, 0-3 \u2014 enter as reported)",
    },
    "liver_disease": {
        "Alamine_Aminotransferase": "Alamine Aminotransferase (ALT)",
        "Aspartate_Aminotransferase": "Aspartate Aminotransferase (AST)",
        "Total_Protiens": "Total Proteins",
        "Albumin_and_Globulin_Ratio": "Albumin / Globulin Ratio",
    },
}


def _humanize(name):
    """Turns a raw column name into a readable label: snake_case and
    camelCase both get split into separate words, and all-caps acronyms
    (e.g. "BMI") are preserved instead of being mangled to "Bmi"."""
    s = name.replace("_", " ")
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)      # lower->Upper boundary
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)    # ACRONYMWord -> ACRONYM Word
    words = []
    for w in s.split():
        words.append(w if w.isupper() else (w[:1].upper() + w[1:].lower()))
    return " ".join(words).strip()


def get_feature_label(disease_key, feature):
    override = FEATURE_LABELS.get(disease_key, {}).get(feature)
    return override if override else _humanize(feature)


class ValidationError(ValueError):
    """Raised when submitted input doesn't match what the model was trained on."""
    pass


def _model_dir(disease_key):
    return os.path.join(PROJECT_ROOT, "models", disease_key)


def get_feature_specs(disease_key):
    """Derives per-feature input constraints directly from the training data,
    so the form and validation always match what the model actually saw —
    no separate config to fall out of sync.

    Returns a dict: {feature: {"type": "categorical", "options": [...],
    "option_labels": {value: display text}}} or {"type": "numeric",
    "min": x, "max": y, "integer": bool}.
    """
    if disease_key not in _spec_cache:
        disease_cfg = get_disease(disease_key)
        df = pd.read_csv(os.path.join(PROJECT_ROOT, disease_cfg["data_path"]))
        target_col = disease_cfg["target_column"]
        X = df.drop(columns=[target_col])
        known_labels = CATEGORY_LABELS.get(disease_key, {})

        specs = {}
        for col in X.columns:
            vals = X[col]
            is_int = pd.api.types.is_integer_dtype(vals)
            nunique = vals.nunique()
            label = get_feature_label(disease_key, col)
            if is_int and nunique <= 10:
                options = sorted(int(v) for v in vals.unique())
                labels = known_labels.get(col)
                option_labels = {o: labels[o] for o in options} if labels \
                    else {o: str(o) for o in options}
                specs[col] = {
                    "type": "categorical",
                    "label": label,
                    "options": options,
                    "option_labels": option_labels,
                    "options_display": " / ".join(option_labels[o] for o in options),
                }
            else:
                specs[col] = {
                    "type": "numeric",
                    "label": label,
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "integer": bool(is_int),
                }
        _spec_cache[disease_key] = specs
    return _spec_cache[disease_key]


def _resolve_categorical(raw, spec):
    """Accepts the underlying numeric code (1), the exact text label
    ("Yes (>120 mg/dl)"), or a shortened version of it ("Yes") — since a
    reasonable user typing into a CSV or text box won't always include a
    parenthetical clinical qualifier. Returns the numeric code, or None if
    nothing matches."""
    try:
        value = float(raw)
        if value in spec["options"]:
            return value
    except (TypeError, ValueError):
        pass

    if isinstance(raw, str):
        norm = raw.strip().lower()
        for opt, label in spec["option_labels"].items():
            if label.strip().lower() == norm:
                return float(opt)
        # Fall back to matching the label with any "(...)" qualifier removed
        for opt, label in spec["option_labels"].items():
            core = re.sub(r"\s*\([^)]*\)", "", label).strip().lower()
            if core == norm:
                return float(opt)
    return None


def normalize_input(disease_key, input_data):
    """Validates submitted values against get_feature_specs AND converts
    categorical text labels ("Male") into the numeric codes the model was
    trained on. Returns a clean {feature: float} dict ready for the model.
    Raises ValidationError listing every problem found, rather than
    silently passing bad values (e.g. sex=4, or an unrecognized label)
    into a model that never saw anything but 0/1 during training."""
    specs = get_feature_specs(disease_key)
    problems = []
    cleaned = {}

    for feature, spec in specs.items():
        if feature not in input_data or input_data[feature] is None or input_data[feature] == "":
            problems.append(f"'{feature}' is required")
            continue

        raw = input_data[feature]

        if spec["type"] == "categorical":
            resolved = _resolve_categorical(raw, spec)
            if resolved is None:
                options_str = ", ".join(spec["option_labels"][o] for o in spec["options"])
                problems.append(
                    f"'{feature}' must be one of: {options_str} (got {raw!r})"
                )
                continue
            cleaned[feature] = resolved
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                problems.append(f"'{feature}' must be a number, got {raw!r}")
                continue
            if value < spec["min"] or value > spec["max"]:
                problems.append(
                    f"'{feature}' should be between {spec['min']:g} and "
                    f"{spec['max']:g} (training data range), got {raw!r}"
                )
                continue
            cleaned[feature] = value

    if problems:
        raise ValidationError("; ".join(problems))
    return cleaned


DEFAULT_META = {
    "hybrid_weights": {"ml": 0.5, "dnn": 0.5},
    "applied_threshold": 0.5,
    "recommended_threshold": 0.5,
    "total_samples": None,
}


def _meta_path(disease_key):
    return os.path.join(_model_dir(disease_key), "meta.json")


def load_artifacts(disease_key):
    """Loads and caches all model artifacts for one condition."""
    if disease_key not in _cache:
        model_dir = _model_dir(disease_key)
        ml_model = joblib.load(os.path.join(model_dir, "best_ml_model.pkl"))
        feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        dnn_path = os.path.join(model_dir, "dnn_model.keras")
        dnn_model = load_model(dnn_path) if os.path.exists(dnn_path) else None

        meta = dict(DEFAULT_META)
        meta_path = _meta_path(disease_key)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta.update(json.load(f))

        _cache[disease_key] = {
            "ml_model": ml_model, "dnn_model": dnn_model, "scaler": scaler,
            "feature_columns": feature_columns, "meta": meta,
        }
    c = _cache[disease_key]
    return c["ml_model"], c["dnn_model"], c["scaler"], c["feature_columns"], c["meta"]


def get_meta(disease_key):
    """Public accessor for a condition's metadata (thresholds, sample size,
    hyperparameters, weights) — used by the admin threshold-tuning page and
    the data-confidence indicator."""
    _, _, _, _, meta = load_artifacts(disease_key)
    return meta


def set_applied_threshold(disease_key, new_threshold):
    """Admin action: overrides the decision threshold used at inference
    time (default 0.5) without retraining. Persists to meta.json and
    invalidates the in-memory cache so the change takes effect immediately."""
    new_threshold = float(new_threshold)
    if not (0.0 < new_threshold < 1.0):
        raise ValidationError("Threshold must be between 0 and 1 (exclusive).")

    meta_path = _meta_path(disease_key)
    meta = dict(DEFAULT_META)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta.update(json.load(f))
    meta["applied_threshold"] = new_threshold

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    _cache.pop(disease_key, None)  # force reload with the new threshold
    return meta


def data_confidence(total_samples):
    """A coarse, honest signal about how much training data backed a
    condition's model — small clinical datasets (a few hundred rows)
    produce less stable estimates than the UI's single probability number
    might otherwise imply."""
    if total_samples is None:
        return "unknown"
    if total_samples < 400:
        return "limited"
    if total_samples < 700:
        return "moderate"
    return "adequate"


def list_features(disease_key):
    _, _, _, feature_columns, _ = load_artifacts(disease_key)
    return feature_columns


def prepare_input(input_data, feature_columns):
    df = pd.DataFrame([input_data])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]


def predict(disease_key, input_data, explain=False):
    disease_cfg = get_disease(disease_key)
    cleaned_input = normalize_input(disease_key, input_data)
    ml_model, dnn_model, scaler, feature_columns, meta = load_artifacts(disease_key)
    weights = meta["hybrid_weights"]
    threshold = meta.get("applied_threshold", 0.5)
    X_input = prepare_input(cleaned_input, feature_columns)

    ml_prob = float(ml_model.predict_proba(X_input)[0][1])

    if dnn_model is not None:
        dnn_prob = float(dnn_model.predict(scaler.transform(X_input), verbose=0)[0][0])
        final_probability = weights["ml"] * ml_prob + weights["dnn"] * dnn_prob
    else:
        dnn_prob = None
        final_probability = ml_prob

    final_prediction = 1 if final_probability >= threshold else 0
    label = disease_cfg["positive_label"] if final_prediction == 1 else disease_cfg["negative_label"]

    result = {
        "condition": disease_key,
        "prediction": final_prediction,
        "probability": final_probability,
        "risk_label": label,
        "data_confidence": data_confidence(meta.get("total_samples")),
    }

    if explain:
        result["explanation"] = explain_prediction(disease_key, X_input, ml_model, feature_columns)

    return result


def _extract_row_values(shap_values):
    arr = np.array(shap_values)
    if isinstance(shap_values, list):
        arr = np.array(shap_values[1] if len(shap_values) > 1 else shap_values[0])
        return arr[0]
    if arr.ndim == 3:
        return arr[0, :, -1]
    if arr.ndim == 2:
        return arr[0]
    return arr


def explain_prediction(disease_key, X_input, ml_model, feature_columns):
    """Returns the top features driving this prediction via SHAP, if available."""
    try:
        import shap
        model_dir = _model_dir(disease_key)
        background_path = os.path.join(model_dir, "shap_background.pkl")

        classifier = ml_model.named_steps.get("classifier", ml_model) if hasattr(ml_model, "named_steps") else ml_model
        preprocessor = ml_model.named_steps.get("scaler") if hasattr(ml_model, "named_steps") else None

        background = joblib.load(background_path) if os.path.exists(background_path) else X_input
        background = background[feature_columns]

        if preprocessor is not None:
            X_for_explainer = preprocessor.transform(X_input)
            background_for_explainer = preprocessor.transform(background)
        else:
            X_for_explainer = X_input.values
            background_for_explainer = background.values

        classifier_type = type(classifier).__name__
        if classifier_type in ("RandomForestClassifier", "GradientBoostingClassifier", "DecisionTreeClassifier"):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_for_explainer)
        elif classifier_type == "LogisticRegression":
            explainer = shap.LinearExplainer(classifier, background_for_explainer)
            shap_values = explainer.shap_values(X_for_explainer)
        else:
            summary = shap.kmeans(background_for_explainer, min(10, len(background_for_explainer)))
            explainer = shap.KernelExplainer(classifier.predict_proba, summary)
            shap_values = explainer.shap_values(X_for_explainer, nsamples=100)

        values = _extract_row_values(shap_values)
        contributions = sorted(zip(feature_columns, values), key=lambda x: abs(x[1]), reverse=True)
        max_abs_impact = max((abs(v) for _, v in contributions), default=0)

        specs = get_feature_specs(disease_key)
        results = []
        for f, v in contributions:
            spec = specs.get(f, {})
            raw_value = X_input[f].iloc[0]
            display_value = _display_feature_value(spec, raw_value)
            direction = "increased" if v > 0 else "decreased" if v < 0 else "had little effect on"
            magnitude = _magnitude_word(abs(v), max_abs_impact)
            label = get_feature_label(disease_key, f)

            if direction == "had little effect on":
                sentence = f"{label} ({display_value}) had little effect on the result."
            else:
                sentence = f"{label} ({display_value}) {magnitude} {direction} the risk."

            results.append({
                "feature": label,
                "value": display_value,
                "impact": round(float(v), 4),
                "text": sentence,
            })
        return results
    except Exception as e:
        return {"error": f"Explanation unavailable: {e}"}


def _display_feature_value(spec, raw_value):
    """Turns a raw numeric value back into something readable -- the option
    label for a categorical field (e.g. "Male" instead of 1), or a cleanly
    rounded number otherwise."""
    if spec.get("type") == "categorical":
        try:
            code = int(float(raw_value))
        except (TypeError, ValueError):
            return str(raw_value)
        return spec.get("option_labels", {}).get(code, str(raw_value))
    try:
        num = float(raw_value)
        return str(int(num)) if num == int(num) else f"{num:.1f}"
    except (TypeError, ValueError):
        return str(raw_value)


def _magnitude_word(abs_impact, max_abs_impact):
    """Classifies a factor's influence relative to the strongest factor in
    THIS explanation -- SHAP values aren't comparable in absolute terms
    across conditions or models, so magnitude is judged relative to its
    own explanation, not against a fixed number."""
    if max_abs_impact == 0:
        return "slightly"
    ratio = abs_impact / max_abs_impact
    if ratio >= 0.66:
        return "strongly"
    if ratio >= 0.33:
        return "moderately"
    return "slightly"
