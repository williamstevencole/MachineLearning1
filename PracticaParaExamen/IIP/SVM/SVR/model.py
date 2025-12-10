import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error

RANDOM_STATE = 42
TARGET = "charges"
TEST_SIZE = 0.2

def load_data(file_path:str) -> pd.DataFrame:
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"El archivo no existe en: {file_path}")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        raise e


def eda(df: pd.DataFrame) -> None:
    print(df.info())
    print(df.describe().T)
    print(df.head())

    numeric_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(10, 8))
    plt.title('Matriz de Correlación')
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.show()

    plt.figure(figsize=(10, 4))
    sns.histplot(df[TARGET], bins=30, kde=True)
    plt.title(f'Distribución de {TARGET}')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["bmi"], y=TARGET, data=df)
    plt.title(f'{TARGET} vs BMI')
    plt.show()
    
def split_train_test(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    Y = df[TARGET]
    return train_test_split(
        X, 
        Y,
        test_size=0.2, 
        random_state=RANDOM_STATE
    )

def get_pipeline(X_train: pd.DataFrame) -> ColumnTransformer:
    num_cols = ['age', 'bmi', 'children']
    cat_cols = ['sex', 'smoker', 'region']

    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])


    preprocessor = ColumnTransformer([            
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])


    return preprocessor

def train(X_train, Y_train, pipeline: ColumnTransformer):
    model = SVR()
    full_pipeline = Pipeline(steps=[
        ('preprocessor', pipeline),
        ('regressor', model)
    ])

    param_grid = {
        'regressor__C': [100, 1000, 10000],
        'regressor__kernel': ['linear', 'rbf'],
        'regressor__gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(
        estimator = full_pipeline,
        param_grid=param_grid,
        cv = 3,
        n_jobs=-1,
        scoring = 'r2'
    )

    grid.fit(X_train, Y_train)

    print(f"\n--- Mejores Hiperparámetros: {grid.best_params_} ---")
    return grid.best_estimator_


def evaluate(model, X_test, Y_test):
    Y_pred = model.predict(X_test)

    mse = mean_squared_error(Y_test, Y_pred)
    rmse = root_mean_squared_error(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    print("\n--- Evaluación del Modelo ---")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}")


def main():
    arg_parser = argparse.ArgumentParser(description="SVM Regressor")
    arg_parser.add_argument('--file', type=str, required=True, help='Path to the CSV file')
    args = arg_parser.parse_args()

    df = load_data(args.file)
    eda(df)

    X_train, X_test, Y_train, Y_test = split_train_test(df)
    pipeline = get_pipeline(X_train)
    model = train(X_train, Y_train, pipeline)

    evaluate(model, X_test, Y_test)



if __name__ == "__main__":
    main()