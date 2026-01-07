import os
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from src.config import *

def evaluate_all_models():
    print("[INFO] Evaluating models...")

    X_test = joblib.load(X_TEST_TFIDF_PATH)
    y_test = joblib.load(Y_TEST_PATH)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    models = {
        "Logistic Regression": "logistic_regression.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Linear SVM": "linear_svm.pkl"
    }

    with open(METRICS_PATH, "w") as f:
        for name, file in models.items():
            model = joblib.load(os.path.join(MODELS_DIR, file))
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")
            cm = confusion_matrix(y_test, y_pred)

            # Write to file
            f.write(f"\n{name}\n")
            f.write(f"Accuracy  : {acc:.4f}\n")
            f.write(f"Precision : {precision:.4f}\n")
            f.write(f"Recall    : {recall:.4f}\n")
            f.write(f"F1-score  : {f1:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_test, y_pred))
            f.write("\n" + "-"*60 + "\n")

            # Print to terminal
            print(f"\n{name}")
            print(f"Accuracy  : {acc:.4f}")
            print(f"Precision : {precision:.4f}")
            print(f"Recall    : {recall:.4f}")
            print(f"F1-score  : {f1:.4f}")
