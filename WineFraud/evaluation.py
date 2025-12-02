import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", "wine_fraud.csv")
TARGET = "quality"
RANDOM_STATE = 101
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", "svm_wine_fraud_model.joblib")

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    X = df.drop(columns=[TARGET], axis=1)
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y
    )

    return X_test, y_test

def load_model(file_path: str):
    model = joblib.load(file_path)
    return model

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("="*20 + " Accuracy " + "="*20)
    print(acc)

    cm = confusion_matrix(y_test, y_pred)
    print("="*20 + " Confusion Matrix " + "="*20)
    print(cm)


    print("="*20 + " Classification Report " + "="*20)
    class_report = classification_report(y_test, y_pred)
    print(class_report)


def main():
    X_test, y_test = load_data(CSV_FILE)
    model = load_model(MODEL_FILE)
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()