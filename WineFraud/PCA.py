import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

RANDOM_STATE = 101
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", "wine_fraud.csv")
TARGET = "quality"

def pca(df: pd.DataFrame, target_col: str):

    # Solo variables numéricas para PCA
    X = df.drop(columns=[target_col, "type"])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # Scree plot
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(pca_full.explained_variance_ratio_)+1),
             pca_full.explained_variance_ratio_, marker='o')
    plt.title("Scree Plot")
    plt.xlabel("Component")
    plt.ylabel("Explained Variance Ratio")
    plt.grid()
    plt.show()

    # PCA 2 components
    pca_2 = PCA(n_components=2, random_state=RANDOM_STATE)
    components_2 = pca_2.fit_transform(X_scaled)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=components_2[:,0], y=components_2[:,1], hue=y, alpha=0.6)
    plt.title("PCA - 2 Components")
    plt.xlabel(f"PC1 ({pca_2.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca_2.explained_variance_ratio_[1]:.2%})")
    plt.legend(title=target_col)
    plt.show()

    # PCA 3 components
    pca_3 = PCA(n_components=3, random_state=RANDOM_STATE)
    components_3 = pca_3.fit_transform(X_scaled)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'Fraud': 'red', 'Legit': 'blue'}
    for label in y.unique():
        mask = y == label
        ax.scatter(
            components_3[mask, 0],
            components_3[mask, 1],
            components_3[mask, 2],
            c=colors[label],
            label=label,
            alpha=0.6
        )

    ax.set_xlabel(f"PC1 ({pca_3.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca_3.explained_variance_ratio_[1]:.2%})")
    ax.set_zlabel(f"PC3 ({pca_3.explained_variance_ratio_[2]:.2%})")
    ax.set_title("PCA - 3 Components")
    ax.legend(title=target_col)
    plt.show()

def main():
    df = pd.read_csv(DATA_PATH)

    pca(df, TARGET)

if __name__ == "__main__":
    main()