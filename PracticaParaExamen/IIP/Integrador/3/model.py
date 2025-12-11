import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report

RANDOM_STATE = 42
TARGET = "price_range"

def load_data(file_path: str) -> pd.DataFrame:
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"El archivo no existe en: {file_path}")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error al cargar: {e}")
        raise e

def get_models() -> dict:
    models = {}

    models['Bagging'] = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

    models['Boosting'] = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)
    models['AdaBoost'] = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)

    estimators= [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE))
    ]

    models['Stacking'] = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1
    )

    return models

def train_and_evaluate_models(models: dict, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for name, model in models.items():
        print(f"\n--- Entrenando y evaluando modelo: {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Reporte de Clasificación:")
        print(classification_report(y_test, y_pred))


def main():
    arg_parse = argparse.ArgumentParser(description="Integrator exam 3")
    arg_parse.add_argument('--train-file', type=str, required=True, help='Path to the training CSV file')
    args = arg_parse.parse_args()

    train_df = load_data(args.train_file)
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    models = get_models()
    train_and_evaluate_models(models, X_train, y_train, X_test, y_test)