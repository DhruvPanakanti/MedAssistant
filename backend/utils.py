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


def _model_dir(disease_key):
    return os.path.join("models", disease_key)


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
