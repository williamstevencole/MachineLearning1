import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

pd.set_option('display.max_columns', None)

def load_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"El archivo no existe en: {file_path}")
    return pd.read_csv(file_path)

def eda(df: pd.DataFrame) -> None:
    print("\n--- EDA: Información del Dataset ---")
    print(df.head())
    print(df.info())
    print(df.describe().T)

    vars_to_plot = ['Annual Income (k$)', 'Spending Score (1-100)']
    
    plt.figure(figsize=(10, 4))
    for i, var in enumerate(vars_to_plot, 1):
        plt.subplot(1, 2, i)
        sns.histplot(df[var], bins=20, kde=True)
        plt.title(f'Distribución de {var}')
    plt.tight_layout()
    plt.show()

def get_pipeline() -> Pipeline:
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    return pipeline

def graph_inertia_kmeans(X, max_k=10):
    inertias = []
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(K_range, inertias, 'bx-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo (Elbow Method)')
    plt.grid(True)
    plt.show()

def train_and_visualize(X_scaled, original_df):
    """ 
    Entrena el modelo final con K=5 (basado en el codo típico de este dataset)
    y visualiza los resultados.
    """
    k_optimo = 5
    kmeans = KMeans(n_clusters=k_optimo, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    score = silhouette_score(X_scaled, labels)
    print(f"\n--- Evaluación del Modelo (K={k_optimo}) ---")
    print(f"Silhouette Score: {score:.4f} (Cercano a 1 es mejor)")

    plot_df = original_df.copy()
    plot_df['Cluster'] = labels
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        hue='Cluster',
        palette='viridis',
        s=100
    )

    plt.title('Segmentación de Clientes (K-Means)')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Script de Clustering para Mall Customers")
    parser.add_argument("--file", type=str, required=True, help="Ruta al archivo CSV")
    args = parser.parse_args()

    df = load_data(args.file)
    eda(df)


    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X_raw = df[features]

    pipeline = get_pipeline()
    X_processed = pipeline.fit_transform(X_raw)

    print("Generando gráfico del codo...")
    graph_inertia_kmeans(X_processed)

    train_and_visualize(X_processed, X_raw)

if __name__ == "__main__":
    main()