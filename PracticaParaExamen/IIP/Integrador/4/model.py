import numpy as np
from pathlib import Path
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

RANDOM_STATE = 42
TARGET = "median_house_value"

def load_data(file_path: str):
    try:
        path = Path(file_path)
        if path.exists():
            df = pd.read_csv(path)
            return df
    except Exception as e:
        print("Error", e)
        raise e
    
def eda(df: pd.DataFrame):
    print(df.info())
    print(df.describe())
    print(df.head(5))

    null_values_pct = df.isna().mean() * 100
    if null_values_pct.sum() > 0:
        df.dropna(inplace=True)
        print("Filas con valores nulos eliminadas.")

def preprocessing(df: pd.DataFrame):
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    return preprocessor

def create_clusters(df: pd.DataFrame, clusters: int = 10):
    kmeans = KMeans(n_clusters=clusters, random_state=RANDOM_STATE)
    
    df['geocluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])
    df['geocluster'] = df['geocluster'].astype('str')

    return df

def get_stacking_model():
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
        ('svr', SVR(kernel='rbf'))
    ]

    model = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=5,
        n_jobs=-1
    )

    return model

def train(preprocessor: ColumnTransformer, model: StackingRegressor, X_train: pd.DataFrame, y_train: pd.Series):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

def evaluation(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = pipeline.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2): {r2}")

    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.show()



def main():
    arg_parse = argparse.ArgumentParser(description="Integrator exercise #4")
    arg_parse.add_argument('--file', type=str, required=True)
    args = arg_parse.parse_args()

    df = load_data(args.file)
    eda(df)
    df = create_clusters(df, clusters=10)

    X = df.drop(columns=[TARGET])
    y = df[TARGET] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    preprocessor = preprocessing(X_train)
    model = get_stacking_model()

    pipeline = train(preprocessor, model, X_train, y_train)
    evaluation(pipeline, X_test, y_test)






if __name__ == "__main__":
    main()