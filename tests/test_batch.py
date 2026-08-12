"""CSV batch prediction: template download and bulk processing."""
import io


def test_template_download_has_correct_headers(client):
    resp = client.get("/heart_disease/batch/template.csv")
    assert resp.status_code == 200
    header = resp.get_data(as_text=True).splitlines()[0]
    assert "sex" in header and "age" in header


def test_batch_predict_mixed_valid_invalid(client):
    csv_content = (
        "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal\n"
        "63,Male,Asymptomatic,145,233,Yes,Normal,150,No,2.3,Upsloping,0 vessels,1\n"
        "50,banana,Typical angina,120,200,No,Normal,170,No,0.5,Upsloping,0 vessels,1\n"
    )
    data = {"file": (io.BytesIO(csv_content.encode()), "test.csv")}
    resp = client.post("/heart_disease/batch", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    rows = resp.get_data(as_text=True).splitlines()
    assert "error" in rows[0]
    assert rows[1].split(",")[-1] == ""
    assert "must be one of" in rows[2]


def test_batch_empty_file_rejected(client):
    data = {"file": (io.BytesIO(b""), "empty.csv")}
    resp = client.post("/heart_disease/batch", data=data,
                        content_type="multipart/form-data", follow_redirects=True)
    assert resp.status_code == 200
    assert b"no data rows" in resp.data.lower() or b"choose a csv" in resp.data.lower()


def test_batch_row_limit_enforced(client):
    header = "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal\n"
    row = "63,Male,Asymptomatic,145,233,Yes,Normal,150,No,2.3,Upsloping,0 vessels,1\n"
    csv_content = header + row * 501
    data = {"file": (io.BytesIO(csv_content.encode()), "big.csv")}
    resp = client.post("/heart_disease/batch", data=data,
                        content_type="multipart/form-data", follow_redirects=True)
    assert b"limit" in resp.data.lower()
