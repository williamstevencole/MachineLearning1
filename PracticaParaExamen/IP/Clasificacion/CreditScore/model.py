import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

RANDOM_STATE = 42
TARGET = "Credit Score"

def load_data() -> pd.DataFrame:
    ds = pd.read_csv("Credit Score Classification Dataset.csv")
    return ds.copy()

def eda(df: pd.DataFrame, target: str = TARGET) -> None:
    print("\nData sample:")
    display(df.head())

    print(f"\nData shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")

    # null percentage
    print("\nNull percentage per column:")
    null_percentage = df.isnull().mean() * 100
    if null_percentage.sum() > 0:
        display(null_percentage[null_percentage > 0])
    else:
        print("No null values found!")

    # target distribution
    print(f"\nTarget distribution ({target}):")
    display(df[target].value_counts())

    # categorical variables
    print("\nCategorical variables:")
    for col in df.select_dtypes(exclude='number').columns:
        if col != target:
            print(f"\n{col}:")
            display(df[col].value_counts()) # Show counts for each category


def split_train_test(df: pd.DataFrame, test_size: float = 0.2):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y
    )

def get_preprocessing_pipeline() -> ColumnTransformer:
    """
    Create preprocessing pipeline with specific encodings:
    - Numeric columns (Age, Income, Number of Children): StandardScaler
    - Education: OrdinalEncoder (ordinal variable)
    - Gender, Marital Status, Home Ownership: OneHotEncoder
    """
    numeric_cols = ['Age', 'Income', 'Number of Children']

    # Ordinal encoding for Education (has a natural order)
    education_categories = [['High School Diploma', "Associate's Degree", "Bachelor's Degree",
                            "Master's Degree", 'Doctorate']]

    # Numeric transformer
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Ordinal transformer for Education
    ordinal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=education_categories, handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # OneHot transformer for categorical variables
    onehot_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('ordinal', ordinal_transformer, ['Education']),
        ('onehot', onehot_transformer, ['Gender', 'Marital Status', 'Home Ownership'])
    ])

    return preprocessor





def train_knn(X_train, y_train, preprocessor):
    """Train KNN classifier with GridSearchCV"""
    print("\n" + "="*50)
    print("Training KNN Classifier")
    print("="*50)

    knn = KNeighborsClassifier()
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', knn)
    ])

    param_grid = {
        'classifier__n_neighbors': [3, 5, 7, 9, 11],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__metric': ['euclidean', 'manhattan']
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")

    return grid.best_estimator_

def train_naive_bayes(X_train, y_train, preprocessor):
    """Train Naive Bayes classifier"""
    print("\n" + "="*50)
    print("Training Naive Bayes Classifier")
    print("="*50)

    nb = GaussianNB()
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', nb)
    ])

    param_grid = {
        'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")

    return grid.best_estimator_

def train_random_forest(X_train, y_train, preprocessor):
    """Train Random Forest classifier with GridSearchCV"""
    print("\n" + "="*50)
    print("Training Random Forest Classifier")
    print("="*50)

    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])

    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")

    return grid.best_estimator_





def evaluate(model, X_test, y_test, model_name: str):
    """Evaluate model with classification report and confusion matrix"""
    print("\n" + "="*50)
    print(f"Evaluation for {model_name}")
    print("="*50)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

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

    return accuracy

def main():
    # Load data
    df = load_data()
    eda(df)

    # Split data (preprocessing will be done in the pipeline)
    X_train, X_test, y_train, y_test = split_train_test(df)

    # Get preprocessing pipeline with custom encodings
    preprocessor = get_preprocessing_pipeline()

    # Train models
    knn_model = train_knn(X_train, y_train, preprocessor)
    naive_bayes_model = train_naive_bayes(X_train, y_train, preprocessor)
    random_forest_model = train_random_forest(X_train, y_train, preprocessor)

    # Evaluate models
    knn_acc = evaluate(knn_model, X_test, y_test, "KNN")
    nb_acc = evaluate(naive_bayes_model, X_test, y_test, "Naive Bayes")
    rf_acc = evaluate(random_forest_model, X_test, y_test, "Random Forest")

    # Summary
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    print(f"KNN Accuracy:          {knn_acc:.4f}")
    print(f"Naive Bayes Accuracy:  {nb_acc:.4f}")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    best_model = max([("KNN", knn_acc), ("Naive Bayes", nb_acc), ("Random Forest", rf_acc)], key=lambda x: x[1])
    print(f"\nBest Model: {best_model[0]} with accuracy {best_model[1]:.4f}")

if __name__ == "__main__":
    main()
