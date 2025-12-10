import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

# Imports para PCA y modelado
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import necesario para gráficos 3D
from mpl_toolkits.mplot3d import Axes3D

TARGET = "diagnosis"

def load_data(file_path: str) -> pd.DataFrame:
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"El archivo no existe en: {file_path}")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        raise e
    
def eda(df: pd.DataFrame) -> tuple:
    print("\n--- EDA: Información del Dataset ---")
    
    # Limpieza
    cols_to_drop = [col for col in ['id', 'Unnamed: 32'] if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    print(df.info())
    
    # Separación
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y

def get_pipeline() -> Pipeline:
    """
    Configuramos PCA con 3 componentes.
    Para el gráfico 2D, simplemente ignoraremos el tercero.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=3, random_state=42))
    ])
    return pipeline

def variance_analysis(pipeline: Pipeline):
    pca_model = pipeline.named_steps['pca']
    evr = pca_model.explained_variance_ratio_
    total_var = np.sum(evr) * 100
    
    print("\n--- Análisis de Varianza (3 Componentes) ---")
    print(f"PC1: {evr[0]*100:.2f}%")
    print(f"PC2: {evr[1]*100:.2f}%")
    print(f"PC3: {evr[2]*100:.2f}%")
    print(f"---------------------------")
    print(f"Varianza Total Acumulada: {total_var:.2f}%")

def visualize_2d(X_pca, y):
    """ Gráfico Clásico 2D """
    print("\nGenerando gráfico 2D...")
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Diagnosis'] = y.values
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PC1', y='PC2', hue='Diagnosis', 
        data=pca_df, palette='viridis', alpha=0.7, s=60
    )
    plt.title('PCA 2D - Cáncer de Mama (PC1 vs PC2)')
    plt.xlabel(f'Componente 1')
    plt.ylabel(f'Componente 2')
    plt.grid(True)
    plt.show()

def visualize_3d(X_pca, y):
    """ Gráfico Interactivo 3D """
    print("Generando gráfico 3D...")
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Diagnosis'] = y.values
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    targets = pca_df['Diagnosis'].unique()
    colors = ['#FF0000', '#0000FF'] # Rojo (M) y Azul (B) aprox
    
    for target, color in zip(targets, colors):
        indices = pca_df['Diagnosis'] == target
        ax.scatter(
            pca_df.loc[indices, 'PC1'],
            pca_df.loc[indices, 'PC2'],
            pca_df.loc[indices, 'PC3'],
            c=color, s=50, alpha=0.6, label=f"Diagnosis: {target}"
        )
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('PCA 3D - Cáncer de Mama')
    ax.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="PCA 2D & 3D Visualization")
    parser.add_argument('--file', type=str, required=True, help='Path to CSV')
    args = parser.parse_args()

    # 1. Carga
    df = load_data(args.file)
    X, y = eda(df)
    
    # 2. Pipeline (Calculamos 3 componentes de una vez)
    pipeline = get_pipeline()
    X_pca = pipeline.fit_transform(X)
    
    # 3. Análisis Matemático
    variance_analysis(pipeline)
    
    # 4. Visualizaciones
    visualize_2d(X_pca, y) # Usa columna 0 y 1
    visualize_3d(X_pca, y) # Usa columna 0, 1 y 2

if __name__ == "__main__":
    main()