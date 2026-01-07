from src.data_preprocessing import load_and_preprocess
from src.feature_engineering import create_features
from src.train import train_all_models
from src.evaluate import evaluate_all_models

if __name__ == "__main__":
    print("=== News Classification Pipeline Started ===")
    load_and_preprocess()
    create_features()
    train_all_models()
    evaluate_all_models()
    print("=== Pipeline Completed Successfully ===")
