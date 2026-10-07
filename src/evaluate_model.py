import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from utils import (
    ensure_directories,
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    PROCESSED_DIR,
    load_joblib,
)


def evaluate_model():
    ensure_directories()

    test_path = os.path.join(PROCESSED_DIR, "test_clean.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError("Önce preprocessing çalıştırılmalı. test_clean.csv bulunamadı.")

    test_df = pd.read_csv(test_path)

    model = load_joblib(os.path.join(MODELS_DIR, "rf_model.pkl"))
    target_encoder = load_joblib(os.path.join(MODELS_DIR, "target_encoder.pkl"))
    feature_columns = load_joblib(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    target_col = load_joblib(os.path.join(MODELS_DIR, "target_col.pkl"))

    if target_col in test_df.columns:
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        y_test_enc = target_encoder.transform(y_test)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test_enc, y_pred)
        report = classification_report(y_test_enc, y_pred, target_names=target_encoder.classes_)
        cm = confusion_matrix(y_test_enc, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
        disp.plot()
        plt.title("Confusion Matrix")
        fig_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()

        with open(os.path.join(REPORTS_DIR, "evaluation_report.txt"), "w", encoding="utf-8") as f:
            f.write("=== MODEL EVALUATION REPORT ===\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(cm))

        print("Değerlendirme tamamlandı.")
        print(f"Accuracy: {acc:.4f}")
        print(report)
        print(f"Confusion matrix kaydedildi: {fig_path}")

    else:
        X_test = test_df[feature_columns]
        y_pred = model.predict(X_test)
        pred_labels = target_encoder.inverse_transform(y_pred)

        result_df = test_df.copy()
        result_df["predicted_label"] = pred_labels

        out_path = os.path.join(REPORTS_DIR, "test_predictions.csv")
        result_df.to_csv(out_path, index=False)

        print("Test verisinde gerçek etiket yok.")
        print(f"Tahminler kaydedildi: {out_path}")


if __name__ == "__main__":
    evaluate_model()