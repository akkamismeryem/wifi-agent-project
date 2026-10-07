import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import (
    ensure_directories,
    get_train_path,
    get_test_path,
    safe_drop_unwanted_columns,
    detect_target_column,
    normalize_binary_label,
    get_numeric_and_categorical_columns,
    save_joblib,
    PROCESSED_DIR,
    MODELS_DIR,
)


def load_data():
    train_path = get_train_path()
    test_path = get_test_path()

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train dosyası bulunamadı: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dosyası bulunamadı: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = safe_drop_unwanted_columns(df).copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


def align_test_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
    """
    Test dosyasında target kolon yoksa sorun değil.
    Train'deki feature kolonlarına göre test kolonlarını hizalar.
    """
    train_feature_cols = [c for c in train_df.columns if c != target_col]

    missing_in_test = [c for c in train_feature_cols if c not in test_df.columns]
    extra_in_test = [c for c in test_df.columns if c not in train_feature_cols]

    if missing_in_test:
        raise ValueError(
            f"Test verisinde train feature kolonları eksik: {missing_in_test}"
        )


    if extra_in_test:
        test_df = test_df.drop(columns=extra_in_test)

    test_df = test_df[train_feature_cols]

    return test_df


def encode_categorical_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
    train_df = train_df.copy()
    test_df = test_df.copy()

    feature_cols, _, categorical_cols = get_numeric_and_categorical_columns(train_df, target_col)

    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()

        combined_values = pd.concat(
            [
                train_df[col].astype(str),
                test_df[col].astype(str),
            ],
            axis=0
        )

        le.fit(combined_values)

        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

        label_encoders[col] = le

    return train_df, test_df, label_encoders, feature_cols


def preprocess_pipeline(binary_classification: bool = True):
    ensure_directories()

    train_df, test_df = load_data()
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    target_col = detect_target_column(train_df)

    # Train target işle
    if binary_classification:
        train_df[target_col] = train_df[target_col].apply(normalize_binary_label)

    # Testte target varsa işle
    test_has_target = target_col in test_df.columns
    if test_has_target and binary_classification:
        test_df[target_col] = test_df[target_col].apply(normalize_binary_label)

    # Eğer testte target varsa çıkar, sonra hizala
    if test_has_target:
        test_features_only = test_df.drop(columns=[target_col])
    else:
        test_features_only = test_df.copy()

    test_features_only = align_test_columns(train_df, test_features_only, target_col)

    # Encode
    train_encoded, test_encoded, label_encoders, feature_cols = encode_categorical_columns(
        train_df,
        test_features_only,
        target_col
    )

    train_clean_path = os.path.join(PROCESSED_DIR, "train_clean.csv")
    test_clean_path = os.path.join(PROCESSED_DIR, "test_clean.csv")

    train_encoded.to_csv(train_clean_path, index=False)
    test_encoded.to_csv(test_clean_path, index=False)

    save_joblib(label_encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))
    save_joblib(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))
    save_joblib(target_col, os.path.join(MODELS_DIR, "target_col.pkl"))

    print("Ön işleme tamamlandı.")
    print(f"Target kolon: {target_col}")
    print(f"Test target içeriyor mu?: {test_has_target}")
    print(f"Train temiz veri kaydedildi: {train_clean_path}")
    print(f"Test temiz veri kaydedildi: {test_clean_path}")

    return train_encoded, test_encoded, target_col, test_has_target


if __name__ == "__main__":
    preprocess_pipeline(binary_classification=True)