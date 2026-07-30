"""
Loads trained artifacts for any configured condition and produces hybrid
(ML + DL) predictions with an optional SHAP-based explanation.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

from config_loader import get_disease

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
        "fbs": {0: "No (\u2264120 mg/dl)", 1: "Yes (>120 mg/dl)"},
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
    },
    "liver_disease": {
        "Gender": {0: "Female", 1: "Male"},
    },
}


class ValidationError(ValueError):
    """Raised when submitted input doesn't match what the model was trained on."""
    pass


def _model_dir(disease_key):
    return os.path.join("models", disease_key)


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
        df = pd.read_csv(disease_cfg["data_path"])
        target_col = disease_cfg["target_column"]
        X = df.drop(columns=[target_col])
        known_labels = CATEGORY_LABELS.get(disease_key, {})

        specs = {}
        for col in X.columns:
            vals = X[col]
            is_int = pd.api.types.is_integer_dtype(vals)
            nunique = vals.nunique()
            if is_int and nunique <= 10:
                options = sorted(int(v) for v in vals.unique())
                labels = known_labels.get(col)
                specs[col] = {
                    "type": "categorical",
                    "options": options,
                    "option_labels": {o: labels[o] for o in options} if labels
                                      else {o: str(o) for o in options},
                }
            else:
                specs[col] = {
                    "type": "numeric",
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "integer": bool(is_int),
                }
        _spec_cache[disease_key] = specs
    return _spec_cache[disease_key]


def validate_input(disease_key, input_data):
    """Checks submitted values against get_feature_specs. Raises ValidationError
    listing every problem found, rather than silently passing bad values (e.g.
    sex=4) into a model that never saw anything but 0/1 during training."""
    specs = get_feature_specs(disease_key)
    problems = []

    for feature, spec in specs.items():
        if feature not in input_data or input_data[feature] is None or input_data[feature] == "":
            problems.append(f"'{feature}' is required")
            continue

        raw = input_data[feature]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            problems.append(f"'{feature}' must be a number, got {raw!r}")
            continue

        if spec["type"] == "categorical":
            if value not in spec["options"]:
                options_str = ", ".join(str(o) for o in spec["options"])
                problems.append(
                    f"'{feature}' must be one of [{options_str}], got {raw!r}"
                )
        else:
            if value < spec["min"] or value > spec["max"]:
                problems.append(
                    f"'{feature}' should be between {spec['min']:g} and "
                    f"{spec['max']:g} (training data range), got {raw!r}"
                )

    if problems:
        raise ValidationError("; ".join(problems))


def load_artifacts(disease_key):
    """Loads and caches all model artifacts for one condition."""
    if disease_key not in _cache:
        model_dir = _model_dir(disease_key)
        ml_model = joblib.load(os.path.join(model_dir, "best_ml_model.pkl"))
        feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        dnn_path = os.path.join(model_dir, "dnn_model.keras")
        dnn_model = load_model(dnn_path) if os.path.exists(dnn_path) else None

        weights = {"ml": 0.5, "dnn": 0.5}
        meta_path = os.path.join(model_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                weights = json.load(f).get("hybrid_weights", weights)

        _cache[disease_key] = {
            "ml_model": ml_model, "dnn_model": dnn_model, "scaler": scaler,
            "feature_columns": feature_columns, "weights": weights,
        }
    c = _cache[disease_key]
    return c["ml_model"], c["dnn_model"], c["scaler"], c["feature_columns"], c["weights"]


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
    validate_input(disease_key, input_data)
    ml_model, dnn_model, scaler, feature_columns, weights = load_artifacts(disease_key)
    X_input = prepare_input(input_data, feature_columns)

    ml_prob = float(ml_model.predict_proba(X_input)[0][1])

    if dnn_model is not None:
        dnn_prob = float(dnn_model.predict(scaler.transform(X_input), verbose=0)[0][0])
        final_probability = weights["ml"] * ml_prob + weights["dnn"] * dnn_prob
    else:
        dnn_prob = None
        final_probability = ml_prob

    final_prediction = 1 if final_probability >= 0.5 else 0
    label = disease_cfg["positive_label"] if final_prediction == 1 else disease_cfg["negative_label"]

    result = {
        "condition": disease_key,
        "prediction": final_prediction,
        "probability": final_probability,
        "ml_probability": ml_prob,
        "dnn_probability": dnn_prob,
        "hybrid_weights": weights,
        "risk_label": label,
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
        return [{"feature": f, "impact": round(float(v), 4)} for f, v in contributions]
    except Exception as e:
        return {"error": f"Explanation unavailable: {e}"}
