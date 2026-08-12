"""PDF report generation."""


def test_pdf_report_generated(client, valid_payloads):
    resp = client.post("/heart_disease/predict/pdf", json=valid_payloads["heart_disease"])
    assert resp.status_code == 200
    assert resp.content_type == "application/pdf"
    assert resp.data[:4] == b"%PDF"
    assert len(resp.data) > 500


def test_pdf_report_rejects_invalid_input(client, valid_payloads):
    payload = dict(valid_payloads["heart_disease"], sex="banana")
    resp = client.post("/heart_disease/predict/pdf", json=payload)
    assert resp.status_code == 400
