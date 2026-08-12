"""Admin-adjustable decision threshold."""
from utils import get_meta, set_applied_threshold


def teardown_function(_fn):
    set_applied_threshold("heart_disease", 0.5)


def test_default_threshold_is_half(client):
    meta = get_meta("heart_disease")
    assert meta["applied_threshold"] == 0.5


def test_admin_can_change_threshold(logged_in_client):
    resp = logged_in_client.post("/admin/thresholds/heart_disease",
                                  data={"threshold": "0.7"}, follow_redirects=True)
    assert resp.status_code == 200
    assert get_meta("heart_disease")["applied_threshold"] == 0.7


def test_invalid_threshold_rejected(logged_in_client):
    logged_in_client.post("/admin/thresholds/heart_disease", data={"threshold": "5.0"})
    assert get_meta("heart_disease")["applied_threshold"] != 5.0


def test_non_admin_cannot_change_threshold(client):
    resp = client.post("/admin/thresholds/heart_disease",
                        data={"threshold": "0.9"}, follow_redirects=False)
    assert resp.status_code == 302
    assert get_meta("heart_disease")["applied_threshold"] != 0.9


def test_threshold_changes_prediction_outcome(logged_in_client, valid_payloads):
    payload = valid_payloads["heart_disease"]
    from utils import predict
    baseline = predict("heart_disease", payload)

    set_applied_threshold("heart_disease", 0.99)
    strict = predict("heart_disease", payload)
    assert strict["probability"] == baseline["probability"]
    assert strict["prediction"] <= baseline["prediction"]
