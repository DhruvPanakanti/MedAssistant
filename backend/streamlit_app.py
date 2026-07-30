"""
Streamlit alternative UI — run with: streamlit run streamlit_app.py
Lets you pick a condition, fills in a dynamic form based on that
condition's feature list, and shows the hybrid ML+DNN prediction with
an optional SHAP explanation.
"""
import streamlit as st
from config_loader import load_disease_config
from utils import predict, list_features, get_feature_specs, ValidationError
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
specs = get_feature_specs(disease_key)

with st.form("predict_form"):
    st.subheader(f"{disease_cfg['display_name']} — enter values")
    input_data = {}
    cols = st.columns(2)
    for i, feature in enumerate(features):
        spec = specs[feature]
        with cols[i % 2]:
            if spec["type"] == "categorical":
                input_data[feature] = st.selectbox(
                    feature, options=spec["options"],
                    format_func=lambda v, s=spec: s["option_labels"][v],
                )
            else:
                step = 1.0 if spec["integer"] else 0.01
                input_data[feature] = st.number_input(
                    feature, min_value=spec["min"], max_value=spec["max"],
                    value=spec["min"], step=step, format="%.3f",
                )

    explain = st.checkbox("Show SHAP explanation")
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        result = predict(disease_key, input_data, explain=explain)
    except ValidationError as e:
        st.error(f"Invalid input: {e}")
        st.stop()

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
