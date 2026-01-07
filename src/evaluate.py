import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from src.config import MODEL_PATHS, METRICS_PATH

def evaluate_model():
    results = {}
    
    # Evaluate Logistic Regression model
    lr_model, vectorizer, X_test, y_test = joblib.load(MODEL_PATHS['logistic_regression'])
    X_test_vec = vectorizer.transform(X_test)
    lr_pred = lr_model.predict(X_test_vec)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_cm = confusion_matrix(y_test, lr_pred)
    results['logistic_regression'] = {'accuracy': lr_acc, 'confusion_matrix': lr_cm}
    print(' Logistic Regression evaluation completed')
    print(f'   Accuracy: {lr_acc:.4f}')
    
    # Evaluate Naive Bayes model
    nb_model, vectorizer, X_test, y_test = joblib.load(MODEL_PATHS['naive_bayes'])
    X_test_vec = vectorizer.transform(X_test)
    nb_pred = nb_model.predict(X_test_vec)
    nb_acc = accuracy_score(y_test, nb_pred)
    nb_cm = confusion_matrix(y_test, nb_pred)
    results['naive_bayes'] = {'accuracy': nb_acc, 'confusion_matrix': nb_cm}
    print(' Naive Bayes evaluation completed')
    print(f'   Accuracy: {nb_acc:.4f}')
    
    # Evaluate Linear SVM model
    svm_model, vectorizer, X_test, y_test = joblib.load(MODEL_PATHS['svm'])
    X_test_vec = vectorizer.transform(X_test)
    svm_pred = svm_model.predict(X_test_vec)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_cm = confusion_matrix(y_test, svm_pred)
    results['svm'] = {'accuracy': svm_acc, 'confusion_matrix': svm_cm}
    print(' Linear SVM evaluation completed')
    print(f'   Accuracy: {svm_acc:.4f}')
    
    # Determine best model
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']
    
    # Write results to file
    with open(METRICS_PATH, 'w') as f:
        f.write('=== MODEL EVALUATION RESULTS ===\n\n')
        f.write('Logistic Regression Model:\n')
        f.write(f'Accuracy: {lr_acc}\n')
        f.write(f'Confusion Matrix:\n{lr_cm}\n\n')
        f.write('Naive Bayes Model:\n')
        f.write(f'Accuracy: {nb_acc}\n')
        f.write(f'Confusion Matrix:\n{nb_cm}\n\n')
        f.write('Linear SVM Model:\n')
        f.write(f'Accuracy: {svm_acc}\n')
        f.write(f'Confusion Matrix:\n{svm_cm}\n\n')
        f.write(f'Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})')

    print(f'\n All models evaluated successfully')
    print(f'Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})')

if __name__ == '__main__':
    evaluate_model()
