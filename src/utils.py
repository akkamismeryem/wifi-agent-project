import os
import json
import joblib
import pandas as pd
from typing import Dict, Any, List


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")


def ensure_directories() -> None:
    """Gerekli klasörleri oluşturur."""
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ]:
        os.makedirs(path, exist_ok=True)


def get_train_path() -> str:
    return os.path.join(RAW_DIR, "Train_data.csv")


def get_test_path() -> str:
    return os.path.join(RAW_DIR, "Test_data.csv")


def save_joblib(obj: Any, path: str) -> None:
    joblib.dump(obj, path)


def load_joblib(path: str) -> Any:
    return joblib.load(path)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def detect_target_column(df: pd.DataFrame) -> str:
    candidates = ["class", "label", "attack", "target"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Hedef kolon bulunamadı. Kolonlar: {list(df.columns)}"
    )


def safe_drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:

    unwanted = ["Unnamed: 0"]
    existing = [c for c in unwanted if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    return df


def normalize_binary_label(value: str) -> str:
    """
    class kolonunu binary hale getirmek için:
    normal -> normal
    diğer her şey -> attack
    """
    value = str(value).strip().lower()
    if value == "normal":
        return "normal"
    return "attack"


def get_numeric_and_categorical_columns(df: pd.DataFrame, target_col: str):
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return feature_cols, numeric_cols, categorical_cols