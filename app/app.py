import os
import sys
import pandas as pd
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from utils import MODELS_DIR, PROCESSED_DIR, load_joblib
from agent_logic import calculate_risk_score


st.set_page_config(page_title="Intrusion Detection Agent", layout="wide")
st.title("Network Intrusion Detection + Risk Score Agent")

test_path = os.path.join(PROCESSED_DIR, "test_clean.csv")

if not os.path.exists(test_path):
    st.warning("Önce preprocessing ve training çalıştırmalısın.")
    st.stop()

df = pd.read_csv(test_path)

model = load_joblib(os.path.join(MODELS_DIR, "rf_model.pkl"))
target_encoder = load_joblib(os.path.join(MODELS_DIR, "target_encoder.pkl"))
feature_columns = load_joblib(os.path.join(MODELS_DIR, "feature_columns.pkl"))
target_col = load_joblib(os.path.join(MODELS_DIR, "target_col.pkl"))

test_has_target = target_col in df.columns

st.sidebar.header("Kayıt Seç")
sample_index = st.sidebar.number_input(
    "Satır index",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

row = df.iloc[sample_index]

if test_has_target:
    true_label = row[target_col]
    X_row = row.drop(labels=[target_col])
else:
    true_label = "YOK"
    X_row = row.copy()

X_input = pd.DataFrame([X_row])[feature_columns]
pred_encoded = model.predict(X_input)[0]
pred_label = target_encoder.inverse_transform([pred_encoded])[0]

risk_result = calculate_risk_score(X_row.to_dict())

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Tahmini")
    st.metric("Gerçek etiket", true_label)
    st.metric("Tahmin", pred_label)

with col2:
    st.subheader("Risk Analizi")
    st.metric("Toplam risk", risk_result["total_risk"])
    st.metric("Risk seviyesi", risk_result["risk_level"])
    st.write(f"**Tahmini saldırı türü:** {risk_result['attack_hint']}")

st.markdown("---")
st.subheader("Bileşen Riskleri")

r1, r2, r3 = st.columns(3)
with r1:
    st.metric("Traffic Risk", risk_result["traffic_risk"])
    st.metric("Error Risk", risk_result["error_risk"])
with r2:
    st.metric("Packet Risk", risk_result["packet_risk"])
    st.metric("Content Risk", risk_result["content_risk"])
with r3:
    st.metric("Host Risk", risk_result["host_risk"])
    st.metric("Anomaly Risk", risk_result["anomaly_risk"])

st.markdown("---")
st.subheader("Tüm Risk Sonucu")
st.json(risk_result)

st.markdown("---")
st.subheader("Seçilen Kayıt")
st.dataframe(pd.DataFrame([row]), use_container_width=True)