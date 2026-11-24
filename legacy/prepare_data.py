import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Optional
import os


def load_csv_data(filepath: str, max_rows: int = 10000) -> pd.DataFrame:
    """
    Load CSV data from the given path.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
    max_rows : int, default=10000
        Maximum number of rows to sample
        
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    original_rows = len(df)
    
    if original_rows > max_rows:
        df = df.sample(n=max_rows, random_state=42)
        print(f"Sampled {max_rows} rows from {original_rows} total rows")
    
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {filepath}")
    return df


def split_data(
    df: pd.DataFrame, 
    target_column: str, 
    test_size: float = 0.2, 
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[LabelEncoder]]:
    """
    Split the pandas DataFrame into training and validation sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to split
    target_column : str
        The column to use as the target variable.
    test_size : float, default=0.2
        Proportion of data to use for testing.
    random_state : int, default=42
        Random seed for reproducibility.
    stratify : bool, default=True
        Whether to use stratification for the split (useful for classification)
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[LabelEncoder]]
        X_train, X_test, y_train, y_test, label_encoder (None if regression)
    """
    print(df.columns)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Check if this is a classification problem
    is_classification = _is_classification_target(y)
    label_encoder = None
    
    # Encode labels for classification problems
    if is_classification:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        print(f"Classification detected. Encoded {len(label_encoder.classes_)} classes: {label_encoder.classes_}")
        y_to_split = y_encoded
    else:
        print("Regression detected. No label encoding applied.")
        y_to_split = y
    
    # Use stratification if requested and target is categorical/classification
    stratify_param = y_to_split if stratify and is_classification else None
    print(f"Stratify param: {stratify_param is not None}")
    
    X_train, X_test, y_train_split, y_test_split = train_test_split(
        X, y_to_split, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_param
    )
    
    # Convert back to Series with proper names
    y_train = pd.Series(y_train_split, index=X_train.index, name=target_column)
    y_test = pd.Series(y_test_split, index=X_test.index, name=target_column)
    
    print(f"Training set - X: {X_train.shape}, y: {y_train.shape}")
    print(f"Test set - X: {X_test.shape}, y: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, label_encoder


def save_split_data(X_train, X_test, y_train, y_test, output_dir="./split_data"):
    """
    Save split data to CSV files.
    
    Parameters
    ----------
    X_train, X_test, y_train, y_test : DataFrame/Series
        Split data from split_data function
    output_dir : str
        Directory to save the files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine features and targets back for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    test_with_labels_df = pd.concat([X_test, y_test], axis=1)
    test_df = X_test  # Just features, no labels
    
    train_path = os.path.join(output_dir, "train.csv")
    test_with_labels_path = os.path.join(output_dir, "test_with_labels.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_with_labels_df.to_csv(test_with_labels_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved train split to: {train_path}")
    print(f"Saved test with labels to: {test_with_labels_path}")
    print(f"Saved test (no labels) to: {test_path}")


def _is_classification_target(y: pd.Series) -> bool:
    """
    Determine if target variable is for classification.
    
    Parameters
    ----------
    y : pd.Series
        Target variable
        
    Returns
    -------
    bool
        True if classification, False if regression
    """
    # Check if target is boolean, categorical, or has limited unique values
    if y.dtype == 'bool' or y.dtype == 'object':
        return True
    
    # For numeric targets, check if they appear to be categorical
    unique_values = y.nunique()
    total_values = len(y)
    
    # If less than 20 unique values and they represent less than 5% of total, likely classification
    if unique_values <= 20 and unique_values / total_values < 0.05:
        return True
    
    return False

if __name__ == "__main__":
    # Example usage
    COMPETETION = "spaceship_titanic"
    PATH = f"data/{COMPETETION}/train_orig.csv"
    TARGET_COLUMN = "Transported"
    OUTPUT_DIR = f"data/{COMPETETION}/"
    df = load_csv_data(PATH)
    X_train, X_test, y_train, y_test, label_encoder = split_data(df, TARGET_COLUMN)
    save_split_data(X_train, X_test, y_train, y_test, output_dir=OUTPUT_DIR)
    
    if label_encoder:
        print(f"Label encoder saved. Original classes: {label_encoder.classes_}")
        print("Use label_encoder.inverse_transform() to convert predictions back to original labels")