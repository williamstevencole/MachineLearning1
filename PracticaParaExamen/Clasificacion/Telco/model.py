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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


TARGET = "Churn"
RANDOM_STATE = 42
pd.set_option("display.max_columns", None)


def load_data() -> pd.DataFrame:
    XLS_NAME = "Telco.csv"
    XLS_PATH = Path(__file__).resolve().parent / XLS_NAME
   
    ds = pd.read_csv(XLS_PATH)
    return ds.copy()

def split_train_test(df: pd.DataFrame, test_size: float = 0.2):
    return(train_test_split(
        df.drop(columns=[TARGET]),
        df[TARGET],
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df[TARGET]
    ))

def eda(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    display(df.info())
    display(df.describe(include="all").T) # Transpose for better readability

    df.drop(columns=['customerID'], inplace=True)

    print("\nNull percentage per column:")
    null_percentage = df.isnull().mean() * 100
    display(null_percentage[null_percentage > 0]) # there are no nulls

    print("\nDuplicated values in the dataset:")
    duplicated_count = df.duplicated().sum()
    print(f"Number of duplicated rows: {duplicated_count}") # there are no duplicates

    print("\nMost correlated features with target variable:")
    
    # Encode target variable for correlation analysis
    df_encoded = df.copy()
    df_encoded[target] = df_encoded[target].map({'Yes': 1, 'No': 0})
    
    numeric_corr = df_encoded.select_dtypes(include=np.number).corr(numeric_only=True)
    
    target_corr = numeric_corr[target].drop(target).sort_values(ascending=False, key=abs)
    display(target_corr)   

    plt.title("Correlation Heatmap")
    sns.heatmap(numeric_corr, cmap="coolwarm", cbar=True, linewidths=0.5, linecolor='black')
    plt.show()

    cols_to_plot = target_corr.index.tolist()[:5]  + target_corr.index.tolist()[-5:]
    for col in cols_to_plot:
        plt.figure()
        sns.boxplot(data=df, x=target, y=col)
        plt.title(f"Box plot of {col} by {target}")
        plt.show()

        plt.figure()
        sns.histplot(data=df, x=col, hue=target, kde=True)
        plt.title(f"Histogram of {col} by {target}")
        plt.show()

    return df_encoded

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    numeric_cols = df_clean.select_dtypes(include='number').columns.tolist()
    print(f"Removing outliers using IQR method for {len(numeric_cols)} columns")

    for col in numeric_cols:
        print(f"Processing column: {col}")
        print(f"Initial shape: {df_clean.shape}")
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR

        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        print(f"Column {col}: kept {df_clean.shape[0]} rows after outlier removal")

    return df_clean

def preprocessor()-> ColumnTransformer:
    """
    Define preprocessing for numeric and categorical columns.
    """
    
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    categorical_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    return preprocessor

def train_models(X_train, y_train, preprocessor: ColumnTransformer):
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

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    rf_param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20]
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

    return knn_best_model, rf_best_model, gnb_best_model

def evaluate_model(model, X_test, y_test, model_name: str):
    """Evaluate model and return accuracy for comparison"""
    y_pred = model.predict(X_test)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy

def save_model(model, filename: str):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def main():
    df = load_data()
    df = eda(df)

    df_clean = remove_outliers(df)
    X_train, X_test, y_train, y_test = split_train_test(df_clean)
    
    preprocessor_pipeline = preprocessor()
    knn_model, rf_model, gnb_model = train_models(X_train, y_train, preprocessor_pipeline)
    print("\n" + "="*50)
    print("Models trained successfully with GridSearchCV!")
    print("="*50)

    # Evaluate models
    print("\n" + "="*50)
    print("Evaluating KNN Classifier")
    print("="*50)
    knn_acc = evaluate_model(knn_model, X_test, y_test, "KNN")

    print("\n" + "="*50)
    print("Evaluating Random Forest Classifier")
    print("="*50)
    rf_acc = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    print("\n" + "="*50)
    print("Evaluating Gaussian Naive Bayes Classifier")
    print("="*50)
    gnb_acc = evaluate_model(gnb_model, X_test, y_test, "Naive Bayes")

    # Model comparison summary
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    print(f"KNN Accuracy:           {knn_acc:.4f}")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"Naive Bayes Accuracy:   {gnb_acc:.4f}")

    best_model = max([("KNN", knn_acc), ("Random Forest", rf_acc), ("Naive Bayes", gnb_acc)], key=lambda x: x[1])
    print(f"\nBest Model: {best_model[0]} with accuracy {best_model[1]:.4f}") 





    




if __name__ == "__main__":
    main()