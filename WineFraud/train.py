import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from IPython.display import display
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "files", "wine_fraud.csv")
MODEL_FILE = os.path.join(SCRIPT_DIR, "files", "svm_wine_fraud_model.joblib")
TARGET = "quality"
RANDOM_STATE = 101
TEST_SIZE = 0.1


pd.set_option('display.max_columns', None)

def load_data(file_path)-> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def split_train_test(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y
    )
    

def eda(df: pd.DataFrame, target_col: str) -> None:
    display(df.head(10))

    display(df.info())

    display(df[target_col].value_counts())

    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_col, data=df)
    plt.title(f"Distribution of {target_col}")
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.countplot(x="type", hue=target_col, data=df)
    plt.title(f"Wine type distribution by {target_col}")
    plt.show()

    red_wines = df[df["type"] == "red"]
    white_wines = df[df["type"] == "white"]

    fraud_red = 100 * (red_wines[target_col] == 'Fraud').sum() / len(red_wines)
    fraud_white = 100 * (white_wines[target_col] == 'Fraud').sum() / len(white_wines)
    print(f"Percentage of fraud in red wines: {fraud_red:.2f}%")
    print(f"Percentage of fraud in white wines: {fraud_white:.2f}%")

    df_copy = df.copy()
    df_copy['target_map'] = df_copy[target_col].map({'Fraud': 1, 'Legit': 0})
    numeric_cols = df_copy.select_dtypes(include=np.number).columns.tolist()
    corr = df_copy[numeric_cols].corr()['target_map'].sort_values(ascending=False)

    display(f"Correlation with fraud:")

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_copy[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

def get_pipeline(df: pd.DataFrame) -> Pipeline:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    num_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, numeric_cols),
        ('cat', cat_transformer, categorical_cols)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('svc', SVC(class_weight='balanced', random_state=RANDOM_STATE))
    ])

    return pipeline

def train(X_train: pd.DataFrame, y_train: pd.Series, pipeline: Pipeline):
    param_grid = {
        'svc__C': [0.001, 0.01, 0.1, 0.5, 1],
        'svc__gamma': ['scale', 'auto'],
        'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    grid = GridSearchCV(
        estimator = pipeline,
        param_grid=param_grid,
        cv = 5,
        scoring = "roc_auc", # accuracy, f1
        verbose = 1,
        n_jobs = -1
    )

    grid.fit(X_train, y_train)

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")


    return grid.best_estimator_

def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
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

    
def save_model(model: Pipeline, file_name: str) -> None:
    joblib.dump(model, file_name)
    print(f"Model saved to {file_name}")


def main():
    df = load_data(CSV_FILE)
    if df is None:
        print("Failed to load data")
        return

    eda(df, TARGET)

    svm_pipeline = get_pipeline(df)

    X_train, X_test, y_train, y_test = split_train_test(df)
    best_model = train(X_train, y_train, svm_pipeline)


    evaluate(best_model, X_test, y_test)
    save_model(best_model, MODEL_FILE)

if __name__ == "__main__":
    main()