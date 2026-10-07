import os
import pandas as pd

from utils import (
    MODELS_DIR,
    PROCESSED_DIR,
    load_joblib,
)
from agent_logic import calculate_risk_score, explain_record


def predict_sample(sample_index: int = 0):
    test_path = os.path.join(PROCESSED_DIR, "test_clean.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError("test_clean.csv bulunamadı. Önce preprocessing çalıştır.")

    df = pd.read_csv(test_path)

    model = load_joblib(os.path.join(MODELS_DIR, "rf_model.pkl"))
    target_encoder = load_joblib(os.path.join(MODELS_DIR, "target_encoder.pkl"))
    feature_columns = load_joblib(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    target_col = load_joblib(os.path.join(MODELS_DIR, "target_col.pkl"))

    if sample_index >= len(df):
        raise IndexError(f"sample_index çok büyük. Veri uzunluğu: {len(df)}")

    row = df.iloc[sample_index]

    if target_col in df.columns:
        true_label = row[target_col]
        X_row = row.drop(labels=[target_col])
    else:
        true_label = "YOK"
        X_row = row

    X_row_df = pd.DataFrame([X_row])[feature_columns]
    pred_encoded = model.predict(X_row_df)[0]
    pred_label = target_encoder.inverse_transform([pred_encoded])[0]

    risk_result = calculate_risk_score(X_row.to_dict())
    explanation = explain_record(X_row.to_dict())

    print("=== TEK KAYIT TAHMİN ===")
    print(f"Gerçek etiket   : {true_label}")
    print(f"Tahmin etiket   : {pred_label}")
    print(explanation)

    return {
        "true_label": true_label,
        "pred_label": pred_label,
        "risk": risk_result,
        "explanation": explanation,
    }


if __name__ == "__main__":
    predict_sample(sample_index=0)