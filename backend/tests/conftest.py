"""
Shared pytest fixtures. Run tests from the project root:
    pytest
Tests exercise the app through Flask's test client — no live server or
network calls needed. They assume `models/` already contains trained
artifacts (run `python train.py` first if it doesn't).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(scope="session", autouse=True)
def _quiet_tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@pytest.fixture()
def client():
    from app import app
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


@pytest.fixture()
def logged_in_client(client):
    client.post("/login", data={"username": "admin", "password": "admin123"})
    return client


VALID_PAYLOADS = {
    "diabetes": {
        "Pregnancies": 2, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20,
        "Insulin": 79, "BMI": 28.5, "DiabetesPedigreeFunction": 0.4, "Age": 33,
    },
    "heart_disease": {
        "age": 63, "sex": "Male", "cp": "Asymptomatic", "trestbps": 145, "chol": 233,
        "fbs": "Yes", "restecg": "Normal", "thalach": 150, "exang": "No",
        "oldpeak": 2.3, "slope": "Upsloping", "ca": "0 vessels", "thal": 1,
    },
    "liver_disease": {
        "Age": 65, "Gender": "Female", "Total_Bilirubin": 0.7, "Direct_Bilirubin": 0.1,
        "Alkaline_Phosphotase": 187, "Alamine_Aminotransferase": 16,
        "Aspartate_Aminotransferase": 18, "Total_Protiens": 6.8, "Albumin": 3.3,
        "Albumin_and_Globulin_Ratio": 0.9,
    },
}


@pytest.fixture()
def valid_payloads():
    return VALID_PAYLOADS


@pytest.fixture()
def breast_cancer_payload():
    from utils import get_feature_specs
    specs = get_feature_specs("breast_cancer")
    return {f: (specs[f]["min"] + specs[f]["max"]) / 2 for f in specs}
