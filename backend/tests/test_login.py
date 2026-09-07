"""Login flow, including the redirect-loop bug fix."""


def test_wrong_credentials_rejected(client):
    resp = client.post("/login", data={"username": "admin", "password": "wrong"},
                        follow_redirects=True)
    assert resp.status_code == 200
    assert b"Invalid credentials" in resp.data


def test_correct_login_redirects_to_history(client):
    resp = client.post("/login", data={"username": "admin", "password": "admin123"},
                        follow_redirects=False)
    assert resp.status_code == 302
    assert "/history" in resp.headers["Location"]


def test_login_page_redirects_when_already_logged_in(logged_in_client):
    resp = logged_in_client.get("/login", follow_redirects=False)
    assert resp.status_code == 302
    assert "/history" in resp.headers["Location"]


def test_logout_then_login_shows_form_again(logged_in_client):
    logged_in_client.get("/logout")
    resp = logged_in_client.get("/login")
    assert resp.status_code == 200
    assert b"Username" in resp.data or b"username" in resp.data


def test_history_accessible_once_logged_in(logged_in_client):
    resp = logged_in_client.get("/history")
    assert resp.status_code == 200
