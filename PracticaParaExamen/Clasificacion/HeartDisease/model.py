import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

RANDOM_STATE = 42
TARGET = "TenYearCHD"

def load_data() -> pd.DataFrame:
    ds = pd.read_csv("framingham_heart_disease.csv")
    df = ds.copy()
    return df

def eda(df: pd.DataFrame, target: str = TARGET) -> None:
    print("\nData sample:")
    display(df.head())

    # null percentage
    print("\nNull percentage per column:")
    null_percentage = df.isnull().mean() * 100
    display(null_percentage[null_percentage > 0])

    # correlation with target
    numeric_corr = df.select_dtypes(include='number').corr()
    corr_target = numeric_corr[target].sort_values(ascending=False)
    print("\nCorrelation of features with target:")
    #display(corr_target)    

    top_corr = corr_target.index.tolist()[1:6]
    bottom_corr = corr_target.index.tolist()[-5:]

    variables_to_display = top_corr + bottom_corr
    for var in variables_to_display:
        sns.boxplot(data=df, x=target, y=var)
        sns.boxplot(data=df, x=target, y=var)
        plt.title(f"Box plot of {var} by {target}")
        plt.show()

        # Disable KDE for variables with low variance to avoid singular matrix errors
        try:
            sns.histplot(data=df, x=var, hue=target, kde=True)
        except:
            sns.histplot(data=df, x=var, hue=target, kde=False)
        plt.title(f"Histogram of {var} by {target}")
        plt.show()
        
def remove_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    df_clean = df.copy()

    cols = df.select_dtypes(include='number').skew().sort_values(ascending=False)
    print(f"Removing outliers for numerical columns based on Z-threshold of {z_thresh}.")
    print(f"Columns to process: {cols.index.tolist()}")

    for c in cols.index:
        col_zscore = (df_clean[c] - df_clean[c].mean()) / df_clean[c].std()
        df_clean = df_clean[col_zscore.abs() <= z_thresh]

    print(f"Data shape after outlier removal: {df_clean.shape}")
    return df_clean

def split_train_test(df: pd.DataFrame, test_size: float = 0.2):
    return train_test_split(
        df.drop(columns=[TARGET]),
        df[TARGET],
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True
    )

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(exclude='number').columns

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    return preprocessor

def train(X_train, y_train, preprocessor):
    log = LinearRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', log)
    ])

    param_grid= {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'saga']
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

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")

    TP, FP, FN, TN = confusion_matrix(y_test, y_pred).ravel()
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))




def main():
    df = load_data()
    #eda(df)
    df_clean = remove_outliers(df)
    eda(df_clean)
    X_train, X_test, y_train, y_test = split_train_test(df_clean)
    preprocessor = preprocessing(X_train)
    model = train(X_train, y_train, preprocessor)



    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()