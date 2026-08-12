"""Prediction correctness: valid inputs, text/code equivalence, validation errors."""
import pytest


@pytest.mark.parametrize("condition", ["diabetes", "heart_disease", "liver_disease"])
def test_valid_prediction_succeeds(client, valid_payloads, condition):
    resp = client.post(f"/{condition}/predict", json=valid_payloads[condition])
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0
    assert "risk_label" in data
    assert data["data_confidence"] in ("limited", "moderate", "adequate")


def test_breast_cancer_prediction(client, breast_cancer_payload):
    resp = client.post("/breast_cancer/predict", json=breast_cancer_payload)
    assert resp.status_code == 200


def test_no_internal_model_fields_leak(client, valid_payloads):
    resp = client.post("/diabetes/predict", json=valid_payloads["diabetes"])
    data = resp.get_json()
    for leaked_field in ("ml_probability", "dnn_probability", "hybrid_weights"):
        assert leaked_field not in data


def test_explanation_included_when_requested(client, valid_payloads):
    payload = dict(valid_payloads["heart_disease"], explain=True)
    resp = client.post("/heart_disease/predict", json=payload)
    data = resp.get_json()
    assert "explanation" in data
    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) > 0
    first = data["explanation"][0]
    assert "feature" in first
    assert "impact" in first
    assert "value" in first
    assert "text" in first
    assert first["feature"] in first["text"]
    assert len(first["text"]) > len(first["feature"]) + 10


def test_categorical_text_and_numeric_code_agree(client, valid_payloads):
    text_payload = dict(valid_payloads["heart_disease"])
    code_payload = dict(text_payload, sex=1, cp=3, fbs=1, restecg=0, exang=0, slope=0, ca=0)
    r1 = client.post("/heart_disease/predict", json=text_payload).get_json()
    r2 = client.post("/heart_disease/predict", json=code_payload).get_json()
    assert r1["probability"] == r2["probability"]


def test_feature_labels_are_properly_humanized(client, valid_payloads):
    # Regression test: camelCase/PascalCase column names (BMI,
    # SkinThickness, DiabetesPedigreeFunction) must split into readable
    # words with acronyms preserved -- not mangled by str.title() into
    # "Bmi" / "Skinthickness".
    payload = dict(valid_payloads["diabetes"], explain=True)
    resp = client.post("/diabetes/predict", json=payload)
    labels = [item["feature"] for item in resp.get_json()["explanation"]]
    assert "BMI" in labels
    assert "Skin Thickness" in labels
    assert "Diabetes Pedigree Function" in labels
    assert "Bmi" not in labels
    assert "Skinthickness" not in labels


def test_invalid_categorical_value_rejected(client, valid_payloads):
    payload = dict(valid_payloads["heart_disease"], sex="banana")
    resp = client.post("/heart_disease/predict", json=payload)
    assert resp.status_code == 400
    assert "sex" in resp.get_json()["error"]


def test_out_of_range_numeric_categorical_code_rejected(client, valid_payloads):
    payload = dict(valid_payloads["heart_disease"], sex=4)
    resp = client.post("/heart_disease/predict", json=payload)
    assert resp.status_code == 400


def test_missing_field_rejected(client, valid_payloads):
    payload = dict(valid_payloads["diabetes"])
    del payload["Glucose"]
    resp = client.post("/diabetes/predict", json=payload)
    assert resp.status_code == 400
    assert "Glucose" in resp.get_json()["error"]


def test_wildly_out_of_range_numeric_rejected(client, valid_payloads):
    payload = dict(valid_payloads["diabetes"], Age=9999)
    resp = client.post("/diabetes/predict", json=payload)
    assert resp.status_code == 400
