from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

from IPython.display import display

RANDOM_SEED = 101
plt.rcParams["figure.figsize"] = (8,6)

TARGET = "sale_price"
CSV_NAME = "Used_Car_Price_Prediction.csv"
CSV_PATH = Path(__file__).resolve().parent / CSV_NAME

def load_data_set() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found at {CSV_PATH!s}.\n"
            "Place the file in the same folder as this script or update CSV_NAME/CVS_PATH."
        )
    data = pd.read_csv(CSV_PATH)
    df = data.copy()

    return df


def split_train_test(df: pd.DataFrame, test_size: float = 0.1):
    # Everything but target variable
    X = df.drop(columns=[TARGET])
    # Target variable only
    y = df[TARGET]

    return train_test_split(X, y, test_size = test_size, random_state = RANDOM_SEED, shuffle = True)


def eda(df: pd.DataFrame, target: str = TARGET) -> None:
    # null percentage
    print("\nNull percentage per column:")
    null_percentage = df.isnull().mean() * 100
    display(null_percentage[null_percentage > 0])

    # heat map
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap")
    plt.show()

    numeric_corr = df.select_dtypes(include=np.number).corr(numeric_only=True)
    corr_target = numeric_corr[TARGET].sort_values(ascending=False)
    print("\nTop Correlation with target variable:")
    display(corr_target.head(6))
    print("\nLowest Correlation with target variable:")
    display(corr_target.tail(5))

    top_corr = corr_target.index.tolist()[1:6]
    bottom_corr = corr_target.index.tolist()[-5:]

    # scatter plots for top and bottom correlated features
    for feature in top_corr + bottom_corr:
        plt.figure()
        sns.scatterplot(data=df, x=feature, y=TARGET, alpha=0.5)
        plt.title(f"Scatter plot of {feature} vs {TARGET}")
        plt.show()

    # histograms of variables
    for feature in df.select_dtypes(include=np.number).columns:
        plt.figure()
        sns.histplot(data=df, x=feature)
        plt.title(f"Histogram of {feature}")
        plt.show()
    
def clip_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    df_clean = df.copy()

    cols = df.select_dtypes(include=np.number).skew().sort_values(ascending=False)
    print(f"Clipping outliers for numerical columns based on Z-threshold of {z_thresh}.")
    print(f"Columns to clip: {cols.index.tolist()}")

    for c in cols.index:
        if c not in df.columns or not np.issubdtype(df[c].dtype, np.number):
            continue
        mean, std = df[c].mean(), df[c].std()
        low, high = mean-z_thresh*std, mean+z_thresh*std
        df[c] = df[c].clip(lower=low, upper=high)
    
    print(f"Outliers clipped for columns: {cols}")
    return df_clean


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore',
                                  sparse_output=False,
                                  min_frequency=0.01,
                                  max_categories=20))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    return preprocessor

def train(X_train, y_train, preprocessor):
    model = ElasticNet(max_iter=50000, random_state=RANDOM_SEED)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    param_grid = {
        'model__alpha': [0.01, 0.1, 0.5, 1, 5, 10, 50],
        'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    }

    grid = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print("Best hyperparameters found:")
    print(grid.best_params_)

    return grid.best_estimator_

def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = root_mean_squared_error(y_val, y_pred)

    print(f"Evaluation Results:\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}")

    plt.figure()
    sns.scatterplot(x=y_val, y=y_pred, alpha = 0.6)
    plt.xlabel("Actual sale_price")
    plt.ylabel("Predicted sale_price")
    plt.title("Actual vs Predicted sale_price")
    lims = [min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())]
    plt.plot(lims, lims, color="red")
    plt.show()

def main():
    df = load_data_set()
    eda(df, TARGET)

    df = clip_outliers(df)
    X_train, X_val, y_train, y_val = split_train_test(df, test_size=0.1)

    preprocessor = build_preprocessor(X_train)
    model = train(X_train, y_train, preprocessor)
    evaluate(model, X_val, y_val)

    joblib.dump(model, "used_car_price_model.joblib")
    
if __name__ == "__main__":
    main()