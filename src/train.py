import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from src.config import *

def train_all_models():
    print("[TRAIN] Training models...")

    X_train = joblib.load(X_TRAIN_TFIDF_PATH)
    y_train = joblib.load(Y_TRAIN_PATH)

    os.makedirs(MODELS_DIR, exist_ok=True)

    lr = LogisticRegression(
        max_iter=10000,
        C=2.5,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1
    )

    # Naive Bayes (no max_iter concept)
    nb = MultinomialNB()

    # Linear SVM – max iterations
    svm = LinearSVC(
        C=3.0,
        class_weight="balanced",
        max_iter=10000,
        tol=1e-5,
        dual=False
    )


    lr.fit(X_train, y_train)
    nb.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    joblib.dump(lr, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    joblib.dump(nb, os.path.join(MODELS_DIR, "naive_bayes.pkl"))
    joblib.dump(svm, os.path.join(MODELS_DIR, "linear_svm.pkl"))

    print("[TRAIN] Models saved successfully")
