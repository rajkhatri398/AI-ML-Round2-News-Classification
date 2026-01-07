import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from src.feature_engineering import get_vectorizer
from src.config import PROCESSED_DATA_PATH, MODEL_PATHS, TEST_SIZE, RANDOM_STATE

def train_model():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df['text']
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    vectorizer = get_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    # Train Logistic Regression model
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_vec, y_train)
    joblib.dump((lr_model, vectorizer, X_test, y_test), MODEL_PATHS['logistic_regression'])
    print(' Logistic Regression model trained successfully')

    # Train Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    joblib.dump((nb_model, vectorizer, X_test, y_test), MODEL_PATHS['naive_bayes'])
    print(' Naive Bayes model trained successfully')

    # Train Linear SVM model
    svm_model = LinearSVC(max_iter=2000)
    svm_model.fit(X_train_vec, y_train)
    joblib.dump((svm_model, vectorizer, X_test, y_test), MODEL_PATHS['svm'])
    print(' Linear SVM model trained successfully')

if __name__ == '__main__':
    train_model()
