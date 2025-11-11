import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from pathlib import Path
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


TARGET="income"
RANDOM_STATE = 101
pd.set_option("display.max_columns", None)

def load_data() -> pd.DataFrame:
    XLS_NAME = "adult.csv"
    XLS_PATH = Path(__file__).resolve().parent / XLS_NAME
    ds = pd.read_csv(XLS_PATH)
    return ds.copy()

def split_train_test(df: pd.DataFrame, test_size: float = 0.2):
    X = df.drop(columns=[TARGET])
    Y = df[TARGET]

    return(train_test_split(
        X,
        Y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=Y
    ))

def eda(df: pd.DataFrame, target: str = TARGET) -> None:
    #display(df.info())
    #display(df.describe(include="all").T) # Transpose for better readability

    print("\nNull percentage per column:")
    null_percentage = df.isnull().mean() * 100
    display(null_percentage[null_percentage > 0]) 

    print("\nDuplicated values in the dataset:")
    duplicated_count = df.duplicated().sum()
    print(f"Number of duplicated rows: {duplicated_count}")
    df = df.drop_duplicates()

    print("\nColumn variable counts:")
    for col in df.select_dtypes(exclude='number').columns:
        if col != target:
            print(f"\n{col}:")
            display(df[col].value_counts())

    df_encoded = df.copy()
    df_encoded[target] = df_encoded[target].map({'<=50K': 0, '>50K': 1})

    numeric_corr = df_encoded.select_dtypes(include=np.number).corr(numeric_only=True)
    corr_target = numeric_corr[target].sort_values(ascending=False)

    sns.heatmap(numeric_corr, cmap="coolwarm", cbar=True, linewidths=0.5, linecolor='black')
    plt.title("Correlation Heatmap")
    plt.show()

    top_corr  =  corr_target.head()[1:6]
    bot_corr = corr_target.tail(5)
    
    vars_to_plot = top_corr.index.tolist() + bot_corr.index.tolist()
    for var in vars_to_plot:
        plt.figure()
        sns.boxplot(data=df, x=target, y=var)
        plt.title(f"Box plot of {var} by {target}")
        plt.show()

        plt.figure()
        sns.histplot(data=df, x=var, hue=target, kde=True)
        plt.title(f"Histogram of {var} by {target}")
        plt.show()  
    
    return df_encoded

def remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    numeric_cols = df_clean.select_dtypes(include='number').columns.tolist()
    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    return df_clean

def preprocessor() -> ColumnTransformer:
    nominal_cols = ["sex", "native.country", "occupation", "workclass", "education","marital.status","relationship"]
    ordinal_cols = [ "education.num"]
    numeric_cols = ["age", "fnlwgt", "capital.gain", "capital.loss", "hours.per.week"]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('nom', nominal_transformer, nominal_cols),
        ('ord', ordinal_transformer, ordinal_cols)
    ])

    return preprocessor

def train_models(X_train, y_train, preprocessor: ColumnTransformer):
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'))
    ])

    rf_param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }

    rf_grid_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=5)
    rf_grid_search.fit(X_train, y_train)
    rf_best_model = rf_grid_search.best_estimator_

    gnb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ])

    gnb_grid_search = GridSearchCV(gnb_pipeline, {}, cv=5)
    gnb_grid_search.fit(X_train, y_train)
    gnb_best_model = gnb_grid_search.best_estimator_

    knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier())
    ])

    knn_param_grid = {
        'classifier__n_neighbors': [3, 5, 7],
        'classifier__weights': ['uniform', 'distance']
    }

    knn_grid_search = GridSearchCV(knn_pipeline, knn_param_grid, cv=5)
    knn_grid_search.fit(X_train, y_train)
    knn_best_model = knn_grid_search.best_estimator_

    return rf_best_model, gnb_best_model, knn_best_model

def evaluate_model(models, X_test, y_test, model_names: list):
    """Evaluate model and return accuracy for comparison"""
    acc = []
    for model, model_name in zip(models, model_names):
        print("\n" + "="*50)
        print(f"Evaluating {model_name}")
        print("="*50)

        y_pred = model.predict(X_test)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")

        acc.append((model_name, accuracy))
    return acc


def main():
    df = load_data()

    df_encoded = eda(df)
    df_no_outliers = remove_outliers_iqr(df_encoded)

    X_train, X_test, y_train, y_test = split_train_test(df_no_outliers)

    preprocessor_obj = preprocessor()
    rf_model, gnb_model, knn_model = train_models(X_train, y_train, preprocessor_obj)
    print("\n" + "="*50)
    print("Models trained successfully with GridSearchCV!")
    print("="*50)

    models = [rf_model, gnb_model, knn_model]
    model_names = ["Random Forest", "Gaussian Naive Bayes", "K-Nearest Neighbors"]
    accuracies = evaluate_model(models, X_test, y_test, model_names)

    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    for model_name, accuracy in accuracies:
        print(f"{model_name:<25} Accuracy: {accuracy:.4f}")

    best_model_name, best_accuracy = max(accuracies, key=lambda x: x[1])
    print(f"\nBest Model: {best_model_name} with accuracy {best_accuracy:.4f}")


if __name__ == "__main__":
    main()