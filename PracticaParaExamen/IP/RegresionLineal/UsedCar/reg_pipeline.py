from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from IPython.display import display

RANDOM_SEED = 101
plt.rcParams["figure.figsize"] = (8,6)

TARGET = "target"
CSV_PATH = "Used_Car_Price_Prediction.csv"

def load_data_set() -> pd.DataFrame:
    data = pd.read_csv(CSV_PATH)
    df = data.copy()

    return df

def split_train_test(df: pd.DataFrame, test_size: float = 0.1):
    # Everything but target variable
    X = df.drop(columns=[TARGET])
    # Target variable only
    y = df[TARGET]

    return train_test_split(X, y, test_size = test_size, random_state = RANDOM_SEED, shuffle = True)

def main():
    df = load_data_set()
    X_train, X_test, y_train, y_test = split_train_test(df)

    pd.set_option('display.max_columns', None)  # Show all columns when printing DataFrames

    display(df.head(30))

if __name__ == "__main__":
    main()