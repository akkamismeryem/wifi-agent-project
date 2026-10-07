import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from utils import (
    ensure_directories,
    save_joblib,
    MODELS_DIR,
    REPORTS_DIR,
)
from data_preprocessing import preprocess_pipeline


def train_model():
    ensure_directories()

    train_df, test_df, target_col, test_has_target = preprocess_pipeline(binary_classification=True)

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    y_encoder = LabelEncoder()
    y_enc = y_encoder.fit_transform(y)

    # train içinden validation ayırma
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=y_encoder.classes_)
    cm = confusion_matrix(y_val, y_pred)

    save_joblib(model, os.path.join(MODELS_DIR, "rf_model.pkl"))
    save_joblib(y_encoder, os.path.join(MODELS_DIR, "target_encoder.pkl"))
    save_joblib(X.columns.tolist(), os.path.join(MODELS_DIR, "feature_columns.pkl"))

    with open(os.path.join(REPORTS_DIR, "training_report.txt"), "w", encoding="utf-8") as f:
        f.write("=== MODEL TRAINING REPORT ===\n")
        f.write(f"Validation Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))

    print("Model eğitildi ve kaydedildi.")
    print(f"Validation Accuracy: {acc:.4f}")
    print(report)


if __name__ == "__main__":
    train_model()