import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from src.config import *

def create_features():
    print("[FE] Creating TF-IDF features...")

    X_train = joblib.load(X_TRAIN_PATH)
    X_test = joblib.load(X_TEST_PATH)

    word_vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        sublinear_tf=True
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.85,
        sublinear_tf=True
    )

    X_train_word = word_vectorizer.fit_transform(X_train)
    X_test_word = word_vectorizer.transform(X_test)

    X_train_char = char_vectorizer.fit_transform(X_train)
    X_test_char = char_vectorizer.transform(X_test)

    X_train_tfidf = hstack([X_train_word, X_train_char])
    X_test_tfidf = hstack([X_test_word, X_test_char])

    joblib.dump(X_train_tfidf, X_TRAIN_TFIDF_PATH)
    joblib.dump(X_test_tfidf, X_TEST_TFIDF_PATH)

    print("[FE] TF-IDF completed")
