"""
Multi-condition medical assistant. Flask API + web UI, driven entirely by
config/diseases.json — adding a new condition means adding a config entry
and a dataset, not new routes.

NOTE: the login here is a minimal demo (hardcoded credentials + Flask
session). Replace with real auth before deploying anywhere real.
"""
import os
from datetime import timedelta
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash, abort

from config_loader import load_disease_config, get_disease
from utils import predict, list_features, get_feature_specs, ValidationError
from database import init_db, save_prediction, get_history
from chatbot import generate_reply

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
