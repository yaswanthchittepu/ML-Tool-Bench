import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple, Optional


def read_data(filepath: str) -> pd.DataFrame:
    """
    Read CSV data into a pandas DataFrame.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file to load.
        
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    """
    df = pd.read_csv(filepath)
    return df

# def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
#     """
#     Split the pandas dataframe into training and validation sets.

#     The training set will contain the features and the target (X_train, Y_train)

#     Likewise, the test set will contain the features and the target (X_val, Y_val)
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input DataFrame to split
#     target_column : str
#         The column to use as the target variable.
#     test_size : float, default=0.2
#         Proportion of data to use for testing.
#     random_state : int, default=42
#         Random seed for reproducibility.
        
#     Returns
#     -------
#     Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
#         X_train, X_val, Y_train, Y_val
#     """
#     X = df.drop(target_column, axis=1)
#     y = df[target_column]
#     X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
#     print(f"X_train shape: {X_train.shape}")
#     print(f"X_val shape: {X_val.shape}")
#     print(f"Y_train shape: {Y_train.shape}")
#     print(f"Y_val shape: {Y_val.shape}")
#     return X_train, X_val, Y_train, Y_val