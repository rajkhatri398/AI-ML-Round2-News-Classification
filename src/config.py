import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "bbc_news.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

X_TRAIN_PATH = os.path.join(PROCESSED_DIR, "X_train.pkl")
X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test.pkl")
Y_TRAIN_PATH = os.path.join(PROCESSED_DIR, "y_train.pkl")
Y_TEST_PATH = os.path.join(PROCESSED_DIR, "y_test.pkl")

X_TRAIN_TFIDF_PATH = os.path.join(PROCESSED_DIR, "X_train_tfidf.pkl")
X_TEST_TFIDF_PATH = os.path.join(PROCESSED_DIR, "X_test_tfidf.pkl")

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.txt")

TEST_SIZE = 0.2
RANDOM_STATE = 42
