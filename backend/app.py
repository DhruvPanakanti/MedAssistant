"""
Multi-condition medical assistant. Flask API + web UI, driven entirely by
config/diseases.json — adding a new condition means adding a config entry
and a dataset, not new routes.

NOTE: the login here is a minimal demo (hardcoded credentials + Flask
session). Replace with real auth before deploying anywhere real.
"""
import os
import io
import csv
import json
from datetime import timedelta
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash, abort, Response, send_file

from config_loader import load_disease_config, get_disease, PROJECT_ROOT
from utils import predict, list_features, get_feature_specs, ValidationError, get_meta, set_applied_threshold
from database import init_db, save_prediction, get_history
from chatbot import generate_reply
from pdf_report import generate_prediction_pdf

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
app.permanent_session_lifetime = timedelta(days=7)

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

init_db()


@app.context_processor
def inject_auth_state():
    # Makes `logged_in` available in every template without passing it
    # from every single route, so the nav can show "History" vs
    # "Admin login" correctly on every page.
    return {"logged_in": bool(session.get("logged_in"))}


@app.route("/favicon.ico")
def favicon():
    # Prevents the browser's automatic favicon request from hitting the
    # catch-all /<disease_key> route below and being treated as an
    # unknown condition.
    return "", 204


@app.route("/")
def home():
    conditions = load_disease_config()
    return render_template("home.html", conditions=conditions)


@app.route("/<disease_key>")
def condition_form(disease_key):
    try:
        disease_cfg = get_disease(disease_key)
    except ValueError:
        abort(404)
    features = list_features(disease_key)
    feature_specs = get_feature_specs(disease_key)
    return render_template("form.html", disease_key=disease_key,
                            disease_cfg=disease_cfg, features=features,
                            feature_specs=feature_specs)


@app.route("/<disease_key>/predict", methods=["POST"])
def condition_predict(disease_key):
    try:
        get_disease(disease_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be JSON"}), 400
    explain = bool(data.pop("explain", False))

    try:
        result = predict(disease_key, data, explain=explain)
        save_prediction(disease_key, data, result)
        return jsonify(result)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


BATCH_ROW_LIMIT = 500


@app.route("/<disease_key>/batch")
def batch_form(disease_key):
    try:
        disease_cfg = get_disease(disease_key)
    except ValueError:
        abort(404)
    features = list_features(disease_key)
    return render_template("batch.html", disease_key=disease_key,
                            disease_cfg=disease_cfg, features=features,
                            row_limit=BATCH_ROW_LIMIT)


@app.route("/<disease_key>/batch/template.csv")
def batch_template(disease_key):
    try:
        features = list_features(disease_key)
    except ValueError:
        abort(404)
    output = io.StringIO()
    csv.writer(output).writerow(features)
    return Response(
        output.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={disease_key}_template.csv"}
    )


@app.route("/<disease_key>/batch", methods=["POST"])
def batch_predict(disease_key):
    try:
        get_disease(disease_key)
    except ValueError:
        abort(404)

    file = request.files.get("file")
    if not file or file.filename == "":
        flash("Please choose a CSV file to upload.", "error")
        return redirect(url_for("batch_form", disease_key=disease_key))

    try:
        text_stream = io.StringIO(file.stream.read().decode("utf-8-sig"))
        rows = list(csv.DictReader(text_stream))
    except Exception as e:
        flash(f"Could not read the CSV file: {e}", "error")
        return redirect(url_for("batch_form", disease_key=disease_key))

    if not rows:
        flash("The uploaded CSV had no data rows.", "error")
        return redirect(url_for("batch_form", disease_key=disease_key))

    if len(rows) > BATCH_ROW_LIMIT:
        flash(f"Batch limit is {BATCH_ROW_LIMIT} rows per upload (got {len(rows)}).", "error")
        return redirect(url_for("batch_form", disease_key=disease_key))

    fieldnames = list(rows[0].keys()) + ["prediction", "probability", "risk_label", "error"]
    output_rows = []
    for row in rows:
        row_out = dict(row)
        try:
            result = predict(disease_key, row)
            save_prediction(disease_key, row, result)
            row_out["prediction"] = result["prediction"]
            row_out["probability"] = round(result["probability"], 4)
            row_out["risk_label"] = result["risk_label"]
            row_out["error"] = ""
        except ValidationError as e:
            row_out.update(prediction="", probability="", risk_label="", error=str(e))
        except Exception as e:
            row_out.update(prediction="", probability="", risk_label="", error=f"Unexpected error: {e}")
        output_rows.append(row_out)

    out_stream = io.StringIO()
    writer = csv.DictWriter(out_stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

    return Response(
        out_stream.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={disease_key}_batch_results.csv"}
    )


@app.route("/citations")
def citations():
    with open(os.path.join(PROJECT_ROOT, "data", "citations.json")) as f:
        citation_data = json.load(f)
    conditions = load_disease_config()
    return render_template("citations.html", citation_data=citation_data, conditions=conditions)


@app.route("/<disease_key>/predict/pdf", methods=["POST"])
def condition_predict_pdf(disease_key):
    try:
        disease_cfg = get_disease(disease_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be JSON"}), 400
    data.pop("explain", None)

    try:
        result = predict(disease_key, data, explain=True)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    buffer = generate_prediction_pdf(disease_key, disease_cfg, data, result)
    return send_file(buffer, mimetype="application/pdf", as_attachment=True,
                      download_name=f"{disease_key}_report.pdf")


@app.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html")


@app.route("/chatbot/message", methods=["POST"])
def chatbot_message():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message is required"}), 400
    reply = generate_reply(message)
    return jsonify(reply)


@app.route("/admin/thresholds")
def admin_thresholds():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    conditions = load_disease_config()
    meta_by_condition = {key: get_meta(key) for key in conditions}
    return render_template("admin_thresholds.html", conditions=conditions,
                            meta_by_condition=meta_by_condition)


@app.route("/admin/thresholds/<disease_key>", methods=["POST"])
def admin_set_threshold(disease_key):
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    try:
        get_disease(disease_key)
    except ValueError:
        abort(404)

    raw_value = request.form.get("threshold")
    try:
        set_applied_threshold(disease_key, raw_value)
        flash(f"Threshold updated for {disease_key}.", "success")
    except (ValidationError, TypeError, ValueError):
        flash(f"Invalid threshold value for {disease_key} — must be a number between 0 and 1.", "error")
    return redirect(url_for("admin_thresholds"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("history"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session.permanent = True
            session["logged_in"] = True
            return redirect(url_for("history"))
        flash("Invalid credentials")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("home"))


@app.route("/history")
def history():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    condition_filter = request.args.get("condition")
    records = get_history(limit=100, condition_key=condition_filter)
    conditions = load_disease_config()
    return render_template("history.html", records=records, conditions=conditions,
                            active_filter=condition_filter)


if __name__ == "__main__":
    app.run(debug=True)
