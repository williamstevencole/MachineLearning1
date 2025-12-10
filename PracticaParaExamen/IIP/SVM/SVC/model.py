import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

RANDOM_STATE = 42
TARGET = "Outcome"
TEST_SIZE = 0.2

pd.set_option('display.max_columns', None)

def load_file(file_path: str) -> pd.DataFrame:
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"El archivo no existe en: {file_path}")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        raise e

def split_train_test(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    Y = df[TARGET]
    return train_test_split(
        X, 
        Y,
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=Y
    )


def eda(df: pd.DataFrame) -> None:
    print("\n--- EDA: Información del Dataset ---")
    print(df.head())
    print(df.info())
    print(df.describe().T)

    numeric_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(10, 8))
    plt.title('Matriz de Correlación')
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.countplot(x=TARGET, data=df)
    plt.title('Distribución de la Variable Objetivo')
    plt.show()


def get_pipeline(X_train: pd.DataFrame) -> Pipeline:
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    pipeline = ColumnTransformer([
        ('num', num_pipeline, num_cols)
    ])

    return pipeline

def train(X_train, Y_train, pipeline: ColumnTransformer):
    model = SVC(random_state=RANDOM_STATE)
    full_pipeline = Pipeline(steps=[
        ('preprocessor', pipeline),
        ('classifier', model)
    ])

    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(
        estimator = full_pipeline,
        param_grid=param_grid,
        cv = 5,
        n_jobs=-1,
        scoring = 'accuracy'
    )

    grid.fit(X_train, Y_train)
    print("\n--- Mejores Hiperparámetros ---")
    print(grid.best_params_)
    return grid.best_estimator_

def evaluation(model: GridSearchCV, X_test: pd.DataFrame, Y_test: pd.Series) -> None:
    Y_pred = model.predict(X_test)
    print("\n--- Evaluación del Modelo ---")
    cm = confusion_matrix(Y_test, Y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Matriz de Confusión')
    plt.show()

    class_report = classification_report(Y_test, Y_pred)
    print("Reporte de Clasificación:\n", class_report)

def main():
    arg_parser = argparse.ArgumentParser(description="SVM Classifier for Diabetes Dataset")
    arg_parser.add_argument('--file', type=str, required=True, help='Path to the CSV file')
    args = arg_parser.parse_args()

    df = load_file(args.file)
    eda(df)

    X_train, X_test, Y_train, Y_test = split_train_test(df)
    pipeline = get_pipeline(X_train)   
    model = train(X_train, Y_train, pipeline)
    evaluation(model, X_test, Y_test)

if __name__ == "__main__":
    main()