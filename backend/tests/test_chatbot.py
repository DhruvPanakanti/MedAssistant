"""Rule-based symptom checker: emergency detection, matches, and misses."""


def test_red_flag_returns_urgent(client):
    resp = client.post("/chatbot/message", json={"message": "I have chest pain and difficulty breathing"})
    data = resp.get_json()
    assert data["type"] == "urgent"
    assert "emergency" in data["text"].lower()


def test_common_symptom_returns_matches(client):
    resp = client.post("/chatbot/message", json={"message": "I have a fever and headache"})
    data = resp.get_json()
    assert data["type"] == "matches"
    assert len(data["matches"]) > 0


def test_greeting_handled(client):
    resp = client.post("/chatbot/message", json={"message": "hello"})
    assert resp.get_json()["type"] == "greeting"


def test_gibberish_returns_no_match(client):
    resp = client.post("/chatbot/message", json={"message": "xyzabc123 qwerty"})
    assert resp.get_json()["type"] == "no_match"


def test_empty_message_rejected(client):
    resp = client.post("/chatbot/message", json={"message": ""})
    assert resp.status_code == 400


def test_chatbot_never_names_specific_drugs():
    from chatbot import _load_kb
    kb = _load_kb()
    banned_terms = ["ibuprofen", "acetaminophen", "paracetamol", "aspirin", "mg"]
    for condition in kb["conditions"]:
        advice_lower = condition["advice"].lower()
        for term in banned_terms:
            assert term not in advice_lower, f"{condition['name']} advice mentions '{term}'"
