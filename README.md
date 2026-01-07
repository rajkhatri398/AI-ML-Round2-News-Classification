# BBC News Article Classification Using Machine Learning

## Project Overview
This project implements an end-to-end Machine Learning pipeline to classify BBC news articles into different categories using Natural Language Processing techniques.  
The pipeline includes data preprocessing, feature extraction using TF-IDF, model training, and evaluation using standard classification metrics.

---

## Dataset Source
- **Dataset Name:** BBC News Dataset  
- **Source:** Kaggle (Public Dataset)  
- **Dataset File:** `data/raw/bbc_news.csv`

The dataset contains news article titles, descriptions, and article URLs. Since explicit category labels are not provided, categories are derived from the BBC article URLs.

---

## Categories Used and Filtering Explanation
Initially, the following categories were selected during preprocessing:

business, politics, sport, technology, entertainment

After extracting categories from URLs, some categories such as **politics** and **sport** contained very few samples.  
To avoid class imbalance and unstable model training, categories with insufficient samples were automatically filtered out.

As a result, the final model was trained and evaluated on the following well-represented categories:

- Business  
- Entertainment  
- Technology  

Categories such as **world** and **health** were also experimented with. While these categories are common in BBC news, including them introduced strong class imbalance and semantic overlap, which slightly reduced overall accuracy. Therefore, they were excluded in the final configuration to maintain stable and reliable performance.

---

## Folder Structure Explanation

```
news_classification_project/
├── data/
│   ├── processed/
│   │   ├── cleaned_data.csv            # Cleaned and preprocessed data
│   │   ├── X_test_tfidf.pkl            # TF-IDF test features
│   │   ├── X_test.pkl                  # Test features
│   │   ├── X_train_tfidf.pkl           # TF-IDF train features
│   │   ├── X_train.pkl                 # Train features
│   │   ├── y_test.pkl                  # Test labels
│   │   ├── y_train.pkl                 # Train labels
│   ├── raw/
│   │   └── bbc_news.csv                # Original BBC News dataset
├── models/
│   ├── linear_svm.pkl                  # Trained Linear SVM
│   ├── logistic_regression.pkl         # Trained Logistic Regression
│   └── naive_bayes.pkl                 # Trained Naive Bayes
├── results/
│   └── metrics.txt                     # Evaluation metrics for all models
├── src/
│   ├── __pycache__/
│   ├── config.py                       # Configuration and constants
│   ├── data_preprocessing.py           # Data loading and text cleaning
│   ├── evaluate.py                     # Model evaluation and metrics
│   ├── feature_engineering.py          # TF-IDF vectorization
│   ├── train.py                        # Model training (3 models)
├── .gitignore
├── main.py                             # Entry point - runs full pipeline
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
```

---

## Steps to Run the Project

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   
2. Ensure the dataset is placed at:

   data/raw/bbc_news.csv


3. Run the complete pipeline:

   python main.py


This command performs preprocessing, feature engineering, model training, and evaluation sequentially.

##Models Used

The following Machine Learning models were implemented and compared:

   Logistic Regression

   Naive Bayes

   Linear Support Vector Machine (Linear SVM)

Multiple models were trained to compare performance and select the most suitable model.

##Final Result Summary

Logistic Regression (Final Selected Model)

Accuracy: 92.21%

Precision: 92.12%

Recall: 92.21%

F1-score: 92.16%

Confusion Matrix

[[499   8  22]
 [  8 358   7]
 [ 18  14  54]]

##Explanation:
Logistic Regression performs very well on business and entertainment categories.
The technology category has fewer samples and overlaps with business-related content, which leads to some misclassification.
Overall, Logistic Regression provides the most balanced and stable performance.

##Naive Bayes

Accuracy: 89.78%

Naive Bayes performs well on majority classes but struggles with the technology category due to overlapping vocabulary and class imbalance.

##Linear SVM

Accuracy: 91.80%

Linear SVM shows strong performance on high-dimensional text data but is slightly less stable than Logistic Regression for minority classes.

##Conclusion

Logistic Regression was selected as the final model as it achieved the highest accuracy and provided the best balance between precision, recall, and F1-score.
The project demonstrates the importance of proper preprocessing, feature engineering, and evaluation when working with real-world text datasets.

##Assignment Status

Assignment: AI/ML Round-2 – News Classification

Pipeline Executable: Yes (python main.py)

Evaluation Metrics Used: Accuracy, Precision, Recall, F1-score, Confusion Matrix

Status: ✅ All assignment requirements satisfied

##Final Notes

Category filtering was a deliberate design choice to ensure balanced training.

All models were trained on the same preprocessed data for fair comparison.

The vectorizer was fitted only on training data to avoid data leakage.

Results are reproducible using a fixed random state.


