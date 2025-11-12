from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import argparse

# scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

import joblib

CSV_FILE = "financial_regression.csv"
CSV_PATH = Path(__file__).resolve().parent / CSV_FILE
TARGET = "gold_close"
RANDOM_STATE = 101

pd.set_option("display.max_columns", None)

def load_data() -> pd.DataFrame:
    data = pd.read_csv(CSV_PATH)
    df = data.copy()
    return df

def split_train_test(df: pd.DataFrame, test_size: float = 0.2):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, shuffle=True)


def eda(df: pd.DataFrame, target= TARGET):
    # Describe every variable
    print(df.columns)
    print(df.describe(include="all").T)

    null_pct = df.isna().mean() * 100
    print("\nNull percentage per column:")
    print(null_pct[null_pct > 0])

    numeric_corr = df.select_dtypes(include=np.number).corr(numeric_only=True)
    sns.heatmap(numeric_corr, cmap="coolwarm", cbar=True, linewidths=0.5, linecolor='black')
    plt.title("Correlation Heatmap")
    plt.show()

    corr_target = numeric_corr[TARGET].sort_values(ascending=False)
    top_corr  =  corr_target.head(10)
    bot_corr = corr_target.tail(10)

    print("\nTop Correlation with target variable:")
    print(top_corr)
    print("\nLowest Correlation with target variable:")
    print(bot_corr)

    for col in top_corr + bot_corr:
        plt.figure()
        sns.scatterplot(data=df, x=col, y=TARGET, alpha=0.5)
        sns.histplot(data=df, x=col, alpha=0.3, color='orange', bins=30, kde=True)
        plt.title(f"Scatter plot of {col} vs {TARGET} and histogram of {col}")
        plt.show()

def remove_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    df_clean = df.copy()

    cols = df.select_dtypes(include=np.number).skew().sort_values(ascending=False)
    print(f"Columns to process: {cols.index.tolist()}")

    for c in cols.index:
        if c not in df.columns or not np.issubdtype(df[c].dtype, np.number):
            continue
        mean, std = df[c].mean(), df[c].std()
        z_scores = (df[c] - mean) / std
        df_clean = df_clean[np.abs(z_scores) <= z_thresh]
        print(f"Column '{c}': Removed {len(df) - len(df_clean)} outliers.")

    return df_clean

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categoric_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categoric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categoric_transformer, categoric_cols)
    ])

    return preprocessor


def train(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer):
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet(random_state=RANDOM_STATE))
    ])

    param_grid = {
        'regressor__alpha': [0.1, 1.0, 10.0],
        'regressor__l1_ratio': [0.1, 0.5, 0.9]
    }

    grid_search = GridSearchCV(
                                estimator=model, 
                                param_grid=param_grid, 
                                cv=5, 
                                scoring='neg_mean_squared_error', 
                                n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {-grid_search.best_score_}")

    return grid_search.best_estimator_

def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

def save_model(model: Pipeline, path: str = "gold_price_model.pkl"):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def main():
    parser = argparse.ArgumentParser(description="Gold Price Prediction Model Pipeline")
    parser.add_argument("--action", type=str, choices=["train", "predict"], required=True)
    parser.add_argument("--model-path", type=str, default="gold_price_model.pkl", help="Path to save/load the model")
    args = parser.parse_args()

    if args.action == "predict":
        model = joblib.load(args.model_path)
        print("Model loaded for prediction.")
        return
    elif args.action == "train":
        df = load_data()
        X_train, X_test, y_train, y_test = split_train_test(df)
        eda(df)
        df = remove_outliers(df)
        preprocessor = preprocessing(df)
        model = train(X_train, y_train, preprocessor)
        evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()