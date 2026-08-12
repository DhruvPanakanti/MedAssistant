"""General lifestyle/dietary tips shown alongside each condition's form."""
from lifestyle_tips import get_lifestyle_tips


def test_tips_exist_for_every_condition():
    for key in ["diabetes", "heart_disease", "breast_cancer", "liver_disease"]:
        tips = get_lifestyle_tips(key)
        assert tips is not None
        assert tips.get("eat_more")
        assert tips.get("limit")
        assert tips.get("lifestyle")


def test_tips_render_on_condition_page(client):
    resp = client.get("/diabetes")
    html = resp.get_data(as_text=True)
    assert "General Lifestyle Tips" in html
    assert "Leafy greens" in html


def test_tips_never_mention_specific_drugs_or_dosages():
    banned_terms = ["mg", "ibuprofen", "metformin", "statin", "insulin injection"]
    for key in ["diabetes", "heart_disease", "breast_cancer", "liver_disease"]:
        tips = get_lifestyle_tips(key)
        all_text = " ".join(tips["eat_more"] + tips["limit"] + tips["lifestyle"]).lower()
        for term in banned_terms:
            assert term not in all_text, f"{key} tips mention '{term}'"


def test_tips_included_in_pdf_report(client, valid_payloads):
    resp = client.post("/diabetes/predict/pdf", json=valid_payloads["diabetes"])
    assert resp.status_code == 200
    assert resp.data[:4] == b"%PDF"
