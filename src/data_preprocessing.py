import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from src.config import DATA_PATH, PROCESSED_DATA_PATH

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def preprocess():
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)

    # Use description as text column since it contains the news content
    df['text'] = df['description'].apply(clean_text)
    
    # Extract category from the link (BBC news categories)
    df['category'] = df['link'].str.extract(r'bbc\.com/news/([^/]+)')
    
    # Keep only text and category columns
    df = df[['text', 'category']].dropna()

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(' Data preprocessing completed')

if __name__ == '__main__':
    preprocess()
