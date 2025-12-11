from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

RANDOM_STATE = 42
TARGET = "price_range"

def load_train_data(file_path: str) -> pd.DataFrame:
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"El archivo no existe en: {file_path}")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        raise e
    
def split_train_test(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test

def component_visualization(X: pd.DataFrame, y: pd.Series) -> tuple:
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', pca)
    ])

    X_pca = pipeline.fit_transform(X)
    
    var_exp = pca.explained_variance_ratio_
    print(f"Varianza explicada por los componentes PCA: {var_exp}")
    print(f"Varianza Total Acumulada: {np.sum(var_exp)*100:.2f}%")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        X_pca[:, 0], 
        X_pca[:, 1], 
        X_pca[:, 2], 
        c=y,
        cmap='viridis', 
        s=50,
        alpha=0.6
    )

    ax.set_title('PCA 3D: Distribución Real de Precios')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.colorbar(scatter, label='Rango de Precio (0-3)')
    plt.show()

    return X_pca, pipeline

def train(X_train: pd.DataFrame, y_train: pd.Series) -> SVC:
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE)
    
    param_grid = {
        'C': [0.1, 1, 10],  
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear', 'poly']
    }

    grid_search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    print(f"Mejores hiperparámetros encontrados: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model: SVC, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy en conjunto de prueba: {acc:.4f}")
    print(f"F1 Score en conjunto de prueba: {f1:.4f}")
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Carga y muestra datos de entrenamiento.")
    parser.add_argument('--train_path', type=str, required=True, help='Ruta al archivo CSV.')
    args = parser.parse_args()
    
    train_df = load_train_data(args.train_path)
    
    X_train_raw, X_test_raw, y_train, y_test = split_train_test(train_df)
    
    X_train_pca, pipeline_pca = component_visualization(X_train_raw, y_train)

    model = train(X_train_pca, y_train)

    X_test_pca = pipeline_pca.transform(X_test_raw)

    evaluate_model(model, X_test_pca, y_test)

    
if __name__ == "__main__":
    main()