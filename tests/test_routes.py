"""Basic route health: every page loads, unknown conditions 404 cleanly."""
import pytest

PUBLIC_PAGES = ["/", "/diabetes", "/heart_disease", "/breast_cancer",
                 "/liver_disease", "/chatbot", "/login", "/citations",
                 "/heart_disease/batch"]


@pytest.mark.parametrize("path", PUBLIC_PAGES)
def test_public_pages_load(client, path):
    resp = client.get(path)
    assert resp.status_code == 200


def test_favicon_does_not_error(client):
    resp = client.get("/favicon.ico")
    assert resp.status_code == 204


def test_unknown_condition_form_404s(client):
    resp = client.get("/not_a_real_condition")
    assert resp.status_code == 404


def test_unknown_condition_predict_404s(client):
    resp = client.post("/not_a_real_condition/predict", json={"explain": False})
    assert resp.status_code == 404


def test_history_requires_login(client):
    resp = client.get("/history", follow_redirects=False)
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_admin_thresholds_requires_login(client):
    resp = client.get("/admin/thresholds", follow_redirects=False)
    assert resp.status_code == 302
