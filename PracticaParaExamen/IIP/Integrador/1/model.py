import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, classification_report

RANDOM_STATE = 42
TARGET = "price_range"

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
    
def eda(X: pd.DataFrame) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', pca)
    ])

    X_pca = pipeline.fit_transform(X)
    var_exp = pca.explained_variance_ratio_
    print(f"Varianza explicada por los componentes PCA: {var_exp}")

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(X_pca)


    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(X_pca[:, 0], 
                         X_pca[:, 1], 
                         X_pca[:, 2], 
                         c=clusters, 
                         cmap='viridis', 
                         s=50
                         )

    ax.set_title('PCA 3D con Clustering KMeans')
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.set_zlabel('Componente Principal 3')
    plt.colorbar(scatter, label='Cluster')
    plt.show()



def main():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--file', type=str, required=True, help='Path to CSV')
    args = arg_parse.parse_args()

    df = load_data(args.file)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    eda(X)



if __name__ == "__main__":
    main()