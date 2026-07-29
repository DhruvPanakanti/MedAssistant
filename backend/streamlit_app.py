"""
Streamlit alternative UI — run with: streamlit run streamlit_app.py
Lets you pick a condition, fills in a dynamic form based on that
condition's feature list, and shows the hybrid ML+DNN prediction with
an optional SHAP explanation.
"""
import streamlit as st
from config_loader import load_disease_config
from utils import predict, list_features
from database import init_db, save_prediction, get_history

st.set_page_config(page_title="Medical Assistant", layout="centered")
init_db()

conditions = load_disease_config()

st.title("Medical Assistant (Hybrid ML + DNN)")
disease_key = st.selectbox(
    "Condition",
    options=list(conditions.keys()),
    format_func=lambda k: conditions[k]["display_name"]
)
disease_cfg = conditions[disease_key]
features = list_features(disease_key)

with st.form("predict_form"):
    st.subheader(f"{disease_cfg['display_name']} — enter values")
    input_data = {}
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            input_data[feature] = st.number_input(feature, value=0.0, format="%.3f")

    explain = st.checkbox("Show SHAP explanation")
    submitted = st.form_submit_button("Predict")

if submitted:
    result = predict(disease_key, input_data, explain=explain)
    save_prediction(disease_key, input_data, result)

    st.subheader(f"Result: {result['risk_label']}")
    st.write(f"Hybrid probability: **{result['probability']:.4f}**")
    st.write(f"ML model probability: {result['ml_probability']:.4f}")
    if result.get("dnn_probability") is not None:
        st.write(f"DNN probability: {result['dnn_probability']:.4f}")

    if explain and isinstance(result.get("explanation"), list):
        st.subheader("Top feature contributions (SHAP)")
        for item in result["explanation"][:5]:
            st.write(f"- **{item['feature']}**: {item['impact']}")

st.divider()
if st.checkbox("Show prediction history for this condition"):
    records = get_history(limit=20, condition_key=disease_key)
    st.dataframe(records)
