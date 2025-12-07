import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from pathlib import Path
import joblib
import argparse

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

RANDOM_STATE = 42
target = "species"

pd.set_option("display.max_columns", None)

def load_data_set(location: str) -> pd.DataFrame:
    CSV_PATH = Path(location)
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found at {location}"
        )
    data = pd.read_csv(CSV_PATH)
    df = data.copy()
    return df

def split_train_test(df: pd.DataFrame, test_size: float = 0.25):
    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=RANDOM_STATE, 
        shuffle=True,
        stratify=y
    )

def eda(df: pd.DataFrame, target: str) -> None:
    display(df.head())
    display(df.describe(include="all").T)

    print("description of each numerical variable")
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    display(df[numerical_cols].describe().T)

    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    for col in cat_cols: 
        print(f"\nValue counts for categorical column: {col}")
        display(df[col].value_counts())

    sns.histplot(data=df, x=target, kde=True)
    plt.title(f"distribution of {target}")
    plt.show()

def preprocessing(df: pd.DataFrame) -> tuple[ColumnTransformer, pd.DataFrame]:
    null_pct = df.isna().mean() * 100
    print("\nNull percentage per column:")
    display(null_pct[null_pct > 0])

    # Remove rows with null values as per requirements
    print(f"\nRows before removing nulls: {len(df)}")
    df = df.dropna()
    print(f"Rows after removing nulls: {len(df)}")

    dupe_val = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {dupe_val}")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if target in cat_cols:
        cat_cols.remove(target)
 
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    return preprocessor, df

def train(preprocessor: ColumnTransformer, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    model = LogisticRegression(n_jobs = -1, max_iter = 5000, solver = "saga")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    param_grid = {
        'model__penalty': ['l1', 'l2'],
        'model__C': [0.01, 0.1, 1, 10, 100]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv = 5,
        scoring="accuracy",
        n_jobs = -1,
    )

    grid.fit(X_train, y_train)
    print("Best hyperparameters found:")
    print(grid.best_params_)

    return grid.best_estimator_

def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> int:
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    cr = classification_report(y_test, y_pred, zero_division=0)

    print("="*20 + " Global Summary " + "="*20)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\n" + "="*20 + " Classification Report " + "="*20)
    print("\nClassification Report:\n")
    print(cr)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location-to", required = True, type = str, default="./penguins_size.csv")
    args = parser.parse_args()

    if(not Path(args.location_to).exists()):
        raise FileNotFoundError(f"The path {args.location_to} does not exist")

    df = load_data_set(args.location_to)
    eda(df, target)
    preprocessor, df = preprocessing(df)
    X_train, X_test, y_train, y_test = split_train_test(df)
    model = train(preprocessor, X_train, y_train)
    evaluate(model, X_test, y_test)

    

if __name__ == "__main__":
    main()