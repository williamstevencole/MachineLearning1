import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

X_TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", "X_test.csv")
Y_TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", "y_test.csv")
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", "wine_fraud.csv")
TARGET = "quality"
RANDOM_STATE = 101
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", "svm_wine_fraud_model.joblib")

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    X_test = pd.read_csv(X_TEST_FILE)
    y_test = pd.read_csv(Y_TEST_FILE)
    return X_test, y_test

def load_model(file_path: str):
    model = joblib.load(file_path)
    return model

def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("="*20 + " Accuracy " + "="*20)
    print(acc)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


    print("="*20 + " Classification Report " + "="*20)
    class_report = classification_report(y_test, y_pred)
    print(class_report)


def main():
    X_test, y_test = load_data()
    model = load_model(MODEL_FILE)
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()