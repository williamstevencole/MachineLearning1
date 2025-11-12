import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error


TARGET = "SalePrice"
RANDOM_STATE = 101
pd.set_option("display.max_columns", None)

def load_data() -> pd.DataFrame:
    CSV_NAME = "train.csv"
    CSV_PATH = Path(__file__).resolve().parent / CSV_NAME
    ds = pd.read_csv(CSV_PATH)
    return ds.copy()

def remove_outliers_IQR(df: pd.DataFrame, cols: list, k: float = 1.5) -> pd.DataFrame:
    """Remove outliers from specified columns using the IQR method with safety check"""
    mask = pd.Series(True, index=df.index)

    for col in cols:
        if col not in df.columns or not np.issubdtype(df[col].dropna().dtype, np.number):
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

        mask &= df[col].between(lower_bound, upper_bound) | df[col].isna()

    before = len(df)
    after = mask.sum()
    removed = before - after
    ratio = removed / before if before > 0 else 0

    if ratio > 0.12:
        print(f"WARNING: More than 12% of data would be removed. Skipping outlier removal.")
        return df
    else:
        print(f"Applied outlier removal. Dataset size: {after} rows")
        return df[mask]
    

def eda(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    df = df.drop(columns=['Id'])
    #display(df.info())
    #display(df.describe(include="all").T)  # Transpose for better readability

    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    print("\nCategorical columns: " + ", ".join(cat_cols))
    for col in cat_cols:
        print(f"\n{col} value counts:")
        display(df[col].value_counts())


    print("\nNull percentage per column:")
    null_percentage = df.isnull().mean() * 100
    columns_to_drop = null_percentage[null_percentage > 50].index.tolist()
    if columns_to_drop:
        print(f"\nDropping columns with more than 50% null values: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)

    print("\nDuplicated values in the dataset:")
    display(df.duplicated().sum())

    numeric_corr = df.select_dtypes(include=np.number).corr(numeric_only=True)
    sns.heatmap(numeric_corr, cmap="coolwarm", cbar=True, linewidths=0.5, linecolor='black')
    plt.title("Correlation Heatmap")
    plt.show()

    corr_target = numeric_corr[TARGET].drop(TARGET).sort_values(ascending=False, key=abs)
    top_corr = corr_target.head(5)
    bot_corr = corr_target.tail(5)

    # Store top correlated features for outlier removal
    top_corr_features = top_corr.index.tolist()
    print(f"\nTop 5 features most correlated with {TARGET}:")
    for feat, corr_val in top_corr.items():
        print(f"  {feat}: {corr_val:.3f}")

    for col in top_corr_features + bot_corr.index.tolist():
        plt.figure()
        sns.scatterplot(data=df, x=col, y=TARGET, alpha=0.5)
        plt.title(f"Scatter plot of {col} vs {TARGET}")
        plt.show()

        plt.figure()
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.show()

    return df, top_corr_features

def split_train_test(df: pd.DataFrame, test_size: float = 0.2):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, shuffle=True, stratify=y)

def preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor

def train_model(X_train: pd.DataFrame, y_train: pd.Series, preproc: ColumnTransformer) -> Pipeline:
    model = Pipeline(steps=[
        ('preprocessor', preproc),
        ('regressor', ElasticNet(random_state=RANDOM_STATE, max_iter=10000))
    ])

    param_grid = {
        'regressor__alpha': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'regressor__l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.2f}")

    return grid_search.best_estimator_

def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"\n=== Model Evaluation ===")
    print(f"Test MAE: ${mae:,.2f}")
    print(f"Test MSE: ${mse:,.2f}")
    print(f"Test RMSE: ${rmse:,.2f}")

    # Visualization: Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5)
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("Actual vs Predicted SalePrice")

    # Perfect prediction line
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # Load data
    df = load_data()
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    df, top_corr_features = eda(df)

    df = remove_outliers_IQR(df, top_corr_features)

    X_train, X_test, y_train, y_test = split_train_test(df)
    preproc = preprocessor(X_train)
    model = train_model(X_train, y_train, preproc)

    evaluate_model(model, X_test, y_test)


    joblib.dump(model, "house_price_model.pkl")



if __name__ == "__main__":
    main()
