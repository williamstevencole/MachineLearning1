import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, f1_score, classification_report

RANDOM_STATE = 42
TARGET = "Churn"

def load_data(file_path: str) -> pd.DataFrame:
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"El archivo no existe en: {file_path}")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error al cargar: {e}")
        raise e

def clean_and_prepare(df: pd.DataFrame) -> tuple:
    print("\n--- Limpieza de Datos ---")
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
   
    nulos = df['TotalCharges'].isna().sum()
    print(f"Se encontraron {nulos} valores vacíos en TotalCharges (se imputarán en el pipeline).")

    df[TARGET] = df[TARGET].map({'Yes': 1, 'No': 0})
    
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    return X, y

def get_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), # Por seguridad
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    return preprocessor

def get_models():
    """
    Define los 3 competidores de la batalla.
    """
    models = {}

    # 1. Random Forest
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=100, 
        random_state=RANDOM_STATE
    )

    # 2. Boosting (Gradient Boosting)
    models['Gradient Boosting'] = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    )

    # 3. Stacking
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)),
        ('svc', SVC(probability=True, random_state=RANDOM_STATE))
    ]
    
    models['Stacking Classifier'] = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=3 
    )

    return models

def main():
    parser = argparse.ArgumentParser(description="Ensemble Battle: RF vs GBM vs Stacking")
    parser.add_argument('--file', type=str, required=True, help='Path to CSV')
    args = parser.parse_args()

    df = load_data(args.file)
    X, y = clean_and_prepare(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = get_preprocessor(X_train)

    modelos_dict = get_models()
    
    print("\n" + "="*40)
    print("      COMIENZA LA BATALLA DE MODELOS      ")
    print("="*40)

    results = []

    for name, model in modelos_dict.items():
        print(f"\nEntrenando {name}...")
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        results.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1, '\n\nClassification Report\n': class_report})
        
        print(f"--> Accuracy: {acc:.4f}")
        print(f"--> F1-Score: {f1:.4f}")
        print("Reporte de Clasificación:\n", class_report)

    print("\n" + "="*40)
    print("            TABLA DE POSICIONES           ")
    print("="*40)
    results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
    print(results_df)

if __name__ == "__main__":
    main()