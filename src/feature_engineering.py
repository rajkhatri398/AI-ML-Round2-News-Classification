from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def get_vectorizer():
    return TfidfVectorizer(max_features=5000)
