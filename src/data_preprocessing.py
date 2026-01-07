import os
import re
import pandas as pd
import nltk
import joblib
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from src.config import *

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

def extract_category_from_link(link):
    if not isinstance(link, str):
        return "unknown"
    match = re.search(r"/news/([a-zA-Z\-]+)", link)
    if match:
        return match.group(1).split("-")[0]
    return "unknown"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

def load_and_preprocess():
    print("[DATA] Loading raw data from", RAW_DATA_PATH)

    df = pd.read_csv(RAW_DATA_PATH)
    df.fillna("", inplace=True)

    df["text"] = df["title"] + " " + df["description"]
    df["clean_text"] = df["text"].apply(clean_text)
    df["category"] = df["link"].apply(extract_category_from_link)

    valid_categories = [
    "business",
    "politics",
    "sport",
    "technology",
    "entertainment"
]



    df = df[df["category"].isin(valid_categories)]

    # Remove very small classes
    df = df.groupby("category").filter(lambda x: len(x) > 200)

    X = df["clean_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    joblib.dump(X_train, X_TRAIN_PATH)
    joblib.dump(X_test, X_TEST_PATH)
    joblib.dump(y_train, Y_TRAIN_PATH)
    joblib.dump(y_test, Y_TEST_PATH)

    print("[DATA] Preprocessing completed successfully")
