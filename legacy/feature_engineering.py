"""
Comprehensive Feature Engineering Tools using Pandas
===================================================

This module provides expressive tools for feature engineering that can generate
any feature from the input data using pandas operations.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional, Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import re

# ============================================================================
# BASIC FEATURE CREATION TOOLS
# ============================================================================

def create_numeric_feature(df: pd.DataFrame, name: str, expression: str) -> pd.DataFrame:
    """
    Create a numeric feature using a pandas expression.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    name : str
        Name of the new feature
    expression : str
        Pandas expression (e.g., 'col1 + col2', 'col1 * col2 / col3')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with new feature added
    """
    df_copy = df.copy()
    try:
        df_copy[name] = df_copy.eval(expression)
        print(f"Created numeric feature '{name}' with expression: {expression}")
    except Exception as e:
        print(f"Error creating feature '{name}': {e}")
    return df_copy

def create_categorical_feature(df: pd.DataFrame, name: str, source_column: str, 
                             mapping: Dict[Any, Any]) -> pd.DataFrame:
    """
    Create a categorical feature by mapping values from a source column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    name : str
        Name of the new feature
    source_column : str
        Source column to map from
    mapping : dict
        Mapping dictionary {original_value: new_value}
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with new feature added
    """
    df_copy = df.copy()
    df_copy[name] = df_copy[source_column].map(mapping)
    print(f"Created categorical feature '{name}' from '{source_column}'")
    return df_copy

def create_conditional_feature(df: pd.DataFrame, name: str, condition: str, 
                             true_value: Any, false_value: Any) -> pd.DataFrame:
    """
    Create a feature based on a condition.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    name : str
        Name of the new feature
    condition : str
        Pandas condition expression (e.g., 'col1 > 0', 'col1 == "value"')
    true_value : Any
        Value when condition is True
    false_value : Any
        Value when condition is False
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with new feature added
    """
    df_copy = df.copy()
    try:
        df_copy[name] = np.where(df_copy.eval(condition), true_value, false_value)
        print(f"Created conditional feature '{name}' with condition: {condition}")
    except Exception as e:
        print(f"Error creating conditional feature '{name}': {e}")
    return df_copy

# ============================================================================
# STRING FEATURE ENGINEERING TOOLS
# ============================================================================

def extract_string_pattern(df: pd.DataFrame, name: str, source_column: str, 
                          pattern: str, group: int = 0) -> pd.DataFrame:
    """
    Extract pattern from string column using regex.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    name : str
        Name of the new feature
    source_column : str
        Source string column
    pattern : str
        Regex pattern to extract
    group : int
        Regex group to extract (default: 0)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with new feature added
    """
    df_copy = df.copy()
    if group > 0:
        df_copy[name] = df_copy[source_column].str.extract(pattern, expand=True).iloc[:, group-1]
    else:
        df_copy[name] = df_copy[source_column].str.extract(pattern, expand=False)
    print(f"Extracted pattern '{pattern}' from '{source_column}' to '{name}'")
    return df_copy

def split_string_column(df: pd.DataFrame, name_prefix: str, source_column: str, 
                       delimiter: str, max_splits: int = -1, indices: List[int] = None) -> pd.DataFrame:
    """
    Split string column and create separate features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    name_prefix : str
        Prefix for new feature names
    source_column : str
        Source string column
    delimiter : str
        Delimiter to split on
    max_splits : int
        Maximum number of splits (-1 for all)
    indices : List[int], optional
        Specific indices to extract from split result. If None, extracts all.
        Example: [1] to get only the middle part, [0, 2] to get first and last parts
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with new features added
    """
    df_copy = df.copy()
    split_df = df_copy[source_column].str.split(delimiter, expand=True, n=max_splits)
    
    if indices is None:
        # Extract all parts (original behavior)
        for i, col in enumerate(split_df.columns):
            df_copy[f"{name_prefix}_{i}"] = split_df[col]
        print(f"Split '{source_column}' into {len(split_df.columns)} features with prefix '{name_prefix}'")
    else:
        # Extract only specified indices
        for i, idx in enumerate(indices):
            if idx < len(split_df.columns):
                df_copy[f"{name_prefix}_{idx}"] = split_df.iloc[:, idx]
                print(f"Extracted index {idx} from '{source_column}' as '{name_prefix}_{idx}'")
            else:
                print(f"Warning: Index {idx} is out of range for split result (max index: {len(split_df.columns)-1})")
    
    return df_copy

# def string_length(df: pd.DataFrame, name: str, source_column: str) -> pd.DataFrame:
#     """
#     Create feature with string length.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     name : str
#         Name of the new feature
#     source_column : str
#         Source string column
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with new feature added
#     """
#     df_copy = df.copy()
#     df_copy[name] = df_copy[source_column].str.len()
#     print(f"Created string length feature '{name}' from '{source_column}'")
#     return df_copy

# def string_contains(df: pd.DataFrame, name: str, source_column: str, substring: str) -> pd.DataFrame:
#     """
#     Create boolean feature indicating if string contains substring.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     name : str
#         Name of the new feature
#     source_column : str
#         Source string column
#     substring : str
#         Substring to search for
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with new feature added
#     """
#     df_copy = df.copy()
#     df_copy[name] = df_copy[source_column].str.contains(substring, na=False)
#     print(f"Created contains feature '{name}' for substring '{substring}'")
#     return df_copy

# def string_startswith(df: pd.DataFrame, name: str, source_column: str, prefix: str) -> pd.DataFrame:
#     """
#     Create boolean feature indicating if string starts with prefix.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     name : str
#         Name of the new feature
#     source_column : str
#         Source string column
#     prefix : str
#         Prefix to check for
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with new feature added
#     """
#     df_copy = df.copy()
#     df_copy[name] = df_copy[source_column].str.startswith(prefix, na=False)
#     print(f"Created startswith feature '{name}' for prefix '{prefix}'")
#     return df_copy

# def string_endswith(df: pd.DataFrame, name: str, source_column: str, suffix: str) -> pd.DataFrame:
#     """
#     Create boolean feature indicating if string ends with suffix.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     name : str
#         Name of the new feature
#     source_column : str
#         Source string column
#     suffix : str
#         Suffix to check for
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with new feature added
#     """
#     df_copy = df.copy()
#     df_copy[name] = df_copy[source_column].str.endswith(suffix, na=False)
#     print(f"Created endswith feature '{name}' for suffix '{suffix}'")
#     return df_copy

# ============================================================================
# AGGREGATION AND GROUPING TOOLS
# ============================================================================

def create_group_aggregation(df: pd.DataFrame, name: str, group_column: str, 
                           agg_column: str, agg_func: str) -> pd.DataFrame:
    """
    Create feature by aggregating within groups.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    name : str
        Name of the new feature
    group_column : str
        Column to group by
    agg_column : str
        Column to aggregate
    agg_func : str
        Aggregation function ('count', 'sum', 'mean', 'std', 'min', 'max', etc.)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with new feature added
    """
    df_copy = df.copy()
    agg_result = df_copy.groupby(group_column)[agg_column].agg(agg_func)
    df_copy[name] = df_copy[group_column].map(agg_result)
    print(f"Created group aggregation '{name}' using {agg_func} on '{agg_column}' grouped by '{group_column}'")
    return df_copy

def get_group_aggregation(df: pd.DataFrame, group_column: str, agg_column: str, 
                         agg_func: str) -> pd.DataFrame:
    """
    Get aggregation result without adding it to the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    group_column : str
        Column to group by
    agg_column : str
        Column to aggregate
    agg_func : str
        Aggregation function ('count', 'sum', 'mean', 'std', 'min', 'max', etc.)
    
    Returns:
    --------
    pd.DataFrame
        Aggregation result as a DataFrame with group values as index
    
    Examples:
    ---------
    >>> # Get mean salary by department
    >>> agg_result = get_group_aggregation(df, 'department', 'salary', 'mean')
    >>> print(agg_result)
    >>> #            salary
    >>> # department        
    >>> # IT        55000.0
    >>> # HR        60000.0
    """
    agg_result = df.groupby(group_column)[agg_column].agg(agg_func)
    # Convert Series to DataFrame
    result_df = agg_result.to_frame()
    print(f"Calculated {agg_func} of '{agg_column}' grouped by '{group_column}'")
    print(f"Result shape: {result_df.shape}")
    return result_df

def create_rolling_feature(df: pd.DataFrame, name: str, source_column: str, 
                          window: int, agg_func: str = 'mean') -> pd.DataFrame:
    """
    Create rolling window feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    name : str
        Name of the new feature
    source_column : str
        Source column for rolling calculation
    window : int
        Window size for rolling calculation
    agg_func : str
        Aggregation function for rolling window
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with new feature added
    """
    df_copy = df.copy()
    df_copy[name] = df_copy[source_column].rolling(window=window, min_periods=1).agg(agg_func)
    print(f"Created rolling feature '{name}' with window {window} and {agg_func}")
    return df_copy

def create_lag_feature(df: pd.DataFrame, name: str, source_column: str, lag: int = 1) -> pd.DataFrame:
    """
    Create lagged feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    name : str
        Name of the new feature
    source_column : str
        Source column to lag
    lag : int
        Number of periods to lag (default: 1)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with new feature added
    """
    df_copy = df.copy()
    df_copy[name] = df_copy[source_column].shift(lag)
    print(f"Created lag feature '{name}' with lag {lag}")
    return df_copy

def create_lead_feature(df: pd.DataFrame, name: str, source_column: str, lead: int = 1) -> pd.DataFrame:
    """
    Create leading feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    name : str
        Name of the new feature
    source_column : str
        Source column to lead
    lead : int
        Number of periods to lead (default: 1)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with new feature added
    """
    df_copy = df.copy()
    df_copy[name] = df_copy[source_column].shift(-lead)
    print(f"Created lead feature '{name}' with lead {lead}")
    return df_copy

# ============================================================================
# TIME SERIES FEATURE ENGINEERING TOOLS
# ============================================================================

def extract_datetime_features(df: pd.DataFrame, datetime_column: str, 
                            features: List[str] = None) -> pd.DataFrame:
    """
    Extract datetime features from datetime column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    datetime_column : str
        Name of datetime column
    features : List[str]
        List of features to extract ('year', 'month', 'day', 'hour', 'minute', 
                                   'second', 'dayofweek', 'dayofyear', 'quarter')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with new features added
    """
    df_copy = df.copy()
    if features is None:
        features = ['year', 'month', 'day', 'hour', 'dayofweek', 'quarter']
    
    dt_col = pd.to_datetime(df_copy[datetime_column])
    
    for feature in features:
        if hasattr(dt_col.dt, feature):
            df_copy[f"{datetime_column}_{feature}"] = getattr(dt_col.dt, feature)
            print(f"Extracted {feature} from {datetime_column}")
    
    return df_copy

def create_time_delta(df: pd.DataFrame, name: str, start_column: str, end_column: str, 
                     unit: str = 'D') -> pd.DataFrame:
    """
    Create time delta feature between two datetime columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    name : str
        Name of the new feature
    start_column : str
        Start datetime column
    end_column : str
        End datetime column
    unit : str
        Time unit ('D' for days, 'H' for hours, 'T' for minutes, 'S' for seconds)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with new feature added
    """
    df_copy = df.copy()
    start_dt = pd.to_datetime(df_copy[start_column])
    end_dt = pd.to_datetime(df_copy[end_column])
    df_copy[name] = (end_dt - start_dt).dt.total_seconds()
    
    # Convert to specified unit
    if unit == 'D':
        df_copy[name] = df_copy[name] / (24 * 3600)
    elif unit == 'H':
        df_copy[name] = df_copy[name] / 3600
    elif unit == 'T':
        df_copy[name] = df_copy[name] / 60
    
    print(f"Created time delta feature '{name}' in {unit} units")
    return df_copy

# ============================================================================
# STATISTICAL FEATURE ENGINEERING TOOLS
# ============================================================================

# def create_percentile_feature(df: pd.DataFrame, name: str, source_column: str, 
#                             percentile: float) -> pd.DataFrame:
#     """
#     Create feature indicating if value is above a specific percentile threshold.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     name : str
#         Name of the new feature
#     source_column : str
#         Source column to calculate percentile from
#     percentile : float
#         Percentile threshold (0-100). Values above this percentile get 1, others get 0.
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with new feature added
#     """
#     df_copy = df.copy()
#     threshold = df_copy[source_column].quantile(percentile / 100)
#     df_copy[name] = (df_copy[source_column] > threshold).astype(int)
#     print(f"Created percentile feature '{name}': 1 if above {percentile}th percentile ({threshold:.4f}), 0 otherwise")
#     return df_copy

# def create_zscore_feature(df: pd.DataFrame, name: str, source_column: str) -> pd.DataFrame:
#     """
#     Create z-score normalized feature.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     name : str
#         Name of the new feature
#     source_column : str
#         Source column to normalize
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with new feature added
#     """
#     df_copy = df.copy()
#     mean_val = df_copy[source_column].mean()
#     std_val = df_copy[source_column].std()
#     df_copy[name] = (df_copy[source_column] - mean_val) / std_val
#     print(f"Created z-score feature '{name}' from '{source_column}'")
#     return df_copy

# def create_binned_feature(df: pd.DataFrame, name: str, source_column: str, 
#                          bins: Union[int, List[float]], labels: Union[List[str], bool, None] = False) -> pd.DataFrame:
#     """
#     Create binned feature from continuous variable.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     name : str
#         Name of the new feature
#     source_column : str
#         Source column to bin
#     bins : int or list
#         Number of bins or bin edges
#     labels : list, bool, or None
#         Labels for bins. False for numeric codes, None for intervals, or custom list
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with new feature added
#     """
#     df_copy = df.copy()
#     df_copy[name] = pd.cut(df_copy[source_column], bins=bins, labels=labels, include_lowest=True)
#     print(f"Created binned feature '{name}' with {len(bins)-1 if isinstance(bins, list) else bins} bins")
#     return df_copy

# def create_quantile_feature(df: pd.DataFrame, name: str, source_column: str, 
#                            q: Union[int, List[float]], labels: Union[List[str], bool, None] = False) -> pd.DataFrame:
#     """
#     Create quantile-based feature from continuous variable.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     name : str
#         Name of the new feature
#     source_column : str
#         Source column to create quantiles from
#     q : int or list
#         Number of quantiles or list of quantile boundaries
#     labels : list, bool, or None
#         Labels for quantiles. False for numeric codes, None for intervals, or custom list
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with new feature added
#     """
#     df_copy = df.copy()
#     df_copy[name] = pd.qcut(df_copy[source_column], q=q, duplicates='drop', labels=labels)
#     print(f"Created quantile feature '{name}' with {q} quantiles")
#     return df_copy

# ============================================================================
# INTERACTION FEATURE TOOLS
# ============================================================================

# def create_interaction_features(df: pd.DataFrame, columns: List[str], 
#                               interaction_type: str = 'product') -> pd.DataFrame:
#     """
#     Create interaction features between multiple columns.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     columns : List[str]
#         List of columns to create interactions for
#     interaction_type : str
#         Type of interaction ('product', 'sum', 'difference', 'ratio')
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with new features added
#     """
#     df_copy = df.copy()
#     for i in range(len(columns)):
#         for j in range(i+1, len(columns)):
#             col1, col2 = columns[i], columns[j]
            
#             if interaction_type == 'product':
#                 df_copy[f"{col1}_x_{col2}"] = df_copy[col1] * df_copy[col2]
#             elif interaction_type == 'sum':
#                 df_copy[f"{col1}_plus_{col2}"] = df_copy[col1] + df_copy[col2]
#             elif interaction_type == 'difference':
#                 df_copy[f"{col1}_minus_{col2}"] = df_copy[col1] - df_copy[col2]
#             elif interaction_type == 'ratio':
#                 df_copy[f"{col1}_div_{col2}"] = df_copy[col1] / (df_copy[col2] + 1e-8)
            
#             print(f"Created {interaction_type} interaction between '{col1}' and '{col2}'")
    
#     return df_copy

# def create_polynomial_features(df: pd.DataFrame, column: str, degree: int = 2) -> pd.DataFrame:
#     """
#     Create polynomial features for a column.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     column : str
#         Source column to create polynomial features from
#     degree : int
#         Maximum polynomial degree (default: 2)
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with new features added
#     """
#     df_copy = df.copy()
#     for d in range(2, degree + 1):
#         df_copy[f"{column}_pow_{d}"] = df_copy[column] ** d
#         print(f"Created polynomial feature '{column}_pow_{d}'")
#     return df_copy

# ============================================================================
# CUSTOM FUNCTION APPLIER
# ============================================================================

def apply_custom_function(df: pd.DataFrame, name: str, source_columns: Union[str, List[str]], 
                         func: Callable) -> pd.DataFrame:
    """
    Apply custom function to create feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    name : str
        Name of the new feature
    source_columns : str or List[str]
        Source column(s) to apply function to
    func : Callable
        Function to apply
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with new feature added
    """
    df_copy = df.copy()
    if isinstance(source_columns, str):
        df_copy[name] = func(df_copy[source_columns])
    else:
        df_copy[name] = func(df_copy[source_columns])
    
    print(f"Applied custom function to create feature '{name}'")
    return df_copy

# ============================================================================
# MISSING VALUE HANDLING TOOLS
# ============================================================================

def fillna_with_value(df: pd.DataFrame, columns: Union[str, List[str]], value: Any) -> pd.DataFrame:
    """
    Fill missing values with a specific value.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : str or List[str]
        Column(s) to fill missing values in
    value : Any
        Value to fill missing values with
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values filled
    """
    df_copy = df.copy()
    if isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(value)
            print(f"Filled missing values in '{col}' with {value}")
    
    return df_copy

def fillna_with_median(df: pd.DataFrame, columns: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values with median of the column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : str or List[str], optional
        Column(s) to fill missing values in. If None, fills all numeric columns.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values filled
    """
    df_copy = df.copy()
    
    if columns is None:
        # Fill all numeric columns
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_columns)
    elif isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        if col in df_copy.columns:
            median_val = df_copy[col].median()
            df_copy[col] = df_copy[col].fillna(median_val)
            print(f"Filled missing values in '{col}' with median: {median_val:.4f}")
    
    return df_copy

def fillna_with_mean(df: pd.DataFrame, columns: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values with mean of the column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : str or List[str], optional
        Column(s) to fill missing values in. If None, fills all numeric columns.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values filled
    """
    df_copy = df.copy()
    
    if columns is None:
        # Fill all numeric columns
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_columns)
    elif isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        if col in df_copy.columns:
            mean_val = df_copy[col].mean()
            df_copy[col] = df_copy[col].fillna(mean_val)
            print(f"Filled missing values in '{col}' with mean: {mean_val:.4f}")
    
    return df_copy

def fillna_with_mode(df: pd.DataFrame, columns: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values with mode of the column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : str or List[str], optional
        Column(s) to fill missing values in. If None, fills all categorical columns.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values filled
    """
    df_copy = df.copy()
    
    if columns is None:
        # Fill all categorical columns
        categorical_columns = df_copy.select_dtypes(include=['object', 'category']).columns
        columns = list(categorical_columns)
    elif isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        if col in df_copy.columns:
            mode_val = df_copy[col].mode()[0] if len(df_copy[col].mode()) > 0 else 'Unknown'
            df_copy[col] = df_copy[col].fillna(mode_val)
            print(f"Filled missing values in '{col}' with mode: {mode_val}")
    
    return df_copy

# def fillna_with_forward_fill(df: pd.DataFrame, columns: Union[str, List[str]] = None, 
#                            limit: int = None) -> pd.DataFrame:
#     """
#     Fill missing values using forward fill (ffill).
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     columns : str or List[str], optional
#         Column(s) to fill missing values in. If None, fills all columns.
#     limit : int, optional
#         Maximum number of consecutive fills
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with missing values filled
#     """
#     df_copy = df.copy()
    
#     if columns is None:
#         columns = list(df_copy.columns)
#     elif isinstance(columns, str):
#         columns = [columns]
    
#     for col in columns:
#         if col in df_copy.columns:
#             df_copy[col] = df_copy[col].fillna(method='ffill', limit=limit)
#             print(f"Forward filled missing values in '{col}'")
    
#     return df_copy

# def fillna_with_backward_fill(df: pd.DataFrame, columns: Union[str, List[str]] = None, 
#                             limit: int = None) -> pd.DataFrame:
#     """
#     Fill missing values using backward fill (bfill).
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     columns : str or List[str], optional
#         Column(s) to fill missing values in. If None, fills all columns.
#     limit : int, optional
#         Maximum number of consecutive fills
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with missing values filled
#     """
#     df_copy = df.copy()
    
#     if columns is None:
#         columns = list(df_copy.columns)
#     elif isinstance(columns, str):
#         columns = [columns]
    
#     for col in columns:
#         if col in df_copy.columns:
#             df_copy[col] = df_copy[col].fillna(method='bfill', limit=limit)
#             print(f"Backward filled missing values in '{col}'")
    
#     return df_copy

# def fillna_with_interpolation(df: pd.DataFrame, columns: Union[str, List[str]] = None, 
#                             method: str = 'linear') -> pd.DataFrame:
#     """
#     Fill missing values using interpolation.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     columns : str or List[str], optional
#         Column(s) to fill missing values in. If None, fills all numeric columns.
#     method : str
#         Interpolation method ('linear', 'polynomial', 'spline', 'time', etc.)
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with missing values filled
#     """
#     df_copy = df.copy()
    
#     if columns is None:
#         # Fill all numeric columns
#         numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
#         columns = list(numeric_columns)
#     elif isinstance(columns, str):
#         columns = [columns]
    
#     for col in columns:
#         if col in df_copy.columns:
#             df_copy[col] = df_copy[col].interpolate(method=method)
#             print(f"Interpolated missing values in '{col}' using {method} method")
    
#     return df_copy

# def fillna_with_group_aggregation(df: pd.DataFrame, target_column: str, group_column: str, 
#                                 agg_func: str = 'mean') -> pd.DataFrame:
#     """
#     Fill missing values using group aggregation.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     target_column : str
#         Column with missing values to fill
#     group_column : str
#         Column to group by
#     agg_func : str
#         Aggregation function ('mean', 'median', 'mode', 'min', 'max')
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with missing values filled
#     """
#     df_copy = df.copy()
    
#     # Calculate group statistics
#     if agg_func == 'mode':
#         group_stats = df_copy.groupby(group_column)[target_column].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mode())
#     else:
#         group_stats = df_copy.groupby(group_column)[target_column].agg(agg_func)
    
#     # Fill missing values
#     df_copy[target_column] = df_copy[target_column].fillna(df_copy[group_column].map(group_stats))
    
#     print(f"Filled missing values in '{target_column}' using {agg_func} grouped by '{group_column}'")
#     return df_copy

# def fillna_with_knn(df: pd.DataFrame, columns: Union[str, List[str]] = None, 
#                    n_neighbors: int = 5) -> pd.DataFrame:
#     """
#     Fill missing values using K-Nearest Neighbors imputation.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     columns : str or List[str], optional
#         Column(s) to fill missing values in. If None, fills all numeric columns.
#     n_neighbors : int
#         Number of neighbors to use for imputation
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with missing values filled
#     """
#     try:
#         from sklearn.impute import KNNImputer
#     except ImportError:
#         print("scikit-learn is required for KNN imputation. Please install it first.")
#         return df.copy()
    
#     df_copy = df.copy()
    
#     if columns is None:
#         # Fill all numeric columns
#         numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
#         columns = list(numeric_columns)
#     elif isinstance(columns, str):
#         columns = [columns]
    
#     # Select only numeric columns for KNN imputation
#     numeric_df = df_copy[columns].select_dtypes(include=[np.number])
    
#     if len(numeric_df.columns) > 0:
#         imputer = KNNImputer(n_neighbors=n_neighbors)
#         imputed_values = imputer.fit_transform(numeric_df)
        
#         for i, col in enumerate(numeric_df.columns):
#             df_copy[col] = imputed_values[:, i]
#             print(f"KNN imputed missing values in '{col}' using {n_neighbors} neighbors")
    
#     return df_copy

# def fillna_with_simple_imputation(df: pd.DataFrame, columns: Union[str, List[str]] = None, 
#                                  strategy: str = 'mean') -> pd.DataFrame:
#     """
#     Fill missing values using simple univariate imputation.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     columns : str or List[str], optional
#         Column(s) to fill missing values in. If None, fills all numeric columns.
#     strategy : str
#         Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with missing values filled
#     """
#     try:
#         from sklearn.impute import SimpleImputer
#     except ImportError:
#         print("scikit-learn is required for multivariate imputation. Please install it first.")
#         return df.copy()
    
#     df_copy = df.copy()
    
#     if columns is None:
#         # Fill all numeric columns
#         numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
#         columns = list(numeric_columns)
#     elif isinstance(columns, str):
#         columns = [columns]
    
#     # Select only numeric columns for imputation
#     numeric_df = df_copy[columns].select_dtypes(include=[np.number])
    
#     if len(numeric_df.columns) > 0:
#         imputer = SimpleImputer(strategy=strategy)
#         imputed_values = imputer.fit_transform(numeric_df)
        
#         for i, col in enumerate(numeric_df.columns):
#             df_copy[col] = imputed_values[:, i]
#             print(f"Simple imputed missing values in '{col}' using {strategy} strategy")
    
#     return df_copy

def fillna_with_condition(df: pd.DataFrame, target_column: str, condition: str, 
                         fill_value: Any) -> pd.DataFrame:
    """
    Fill missing values in a column based on a condition.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_column : str
        Column with missing values to fill
    condition : str
        Pandas condition expression (e.g., 'col1 == 1', 'col1 > 0 and col2 == "value"')
    fill_value : Any
        Value to fill missing values with when condition is True
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values filled based on condition
    """
    df_copy = df.copy()
    
    # Create mask for rows where condition is True AND target column is null
    condition_mask = df_copy.eval(condition)
    null_mask = df_copy[target_column].isnull()
    combined_mask = condition_mask & null_mask
    
    # Fill missing values where condition is satisfied
    df_copy.loc[combined_mask, target_column] = fill_value
    
    filled_count = combined_mask.sum()
    print(f"Filled {filled_count} missing values in '{target_column}' where condition '{condition}' is True with value {fill_value}")
    
    return df_copy

def fillna_with_multiple_conditions(df: pd.DataFrame, target_column: str, 
                                  conditions_and_values: List[tuple]) -> pd.DataFrame:
    """
    Fill missing values in a column based on multiple conditions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_column : str
        Column with missing values to fill
    conditions_and_values : List[tuple]
        List of (condition, fill_value) tuples. Conditions are evaluated in order.
        WARNING: Conditions should be mutually exclusive (no overlap), otherwise 
        later conditions will overwrite earlier ones.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values filled based on conditions
    """
    df_copy = df.copy()
    total_filled = 0
    
    for condition, fill_value in conditions_and_values:
        # Create mask for rows where condition is True AND target column is still null
        condition_mask = df_copy.eval(condition)
        null_mask = df_copy[target_column].isnull()
        combined_mask = condition_mask & null_mask
        
        # Fill missing values where condition is satisfied
        df_copy.loc[combined_mask, target_column] = fill_value
        
        filled_count = combined_mask.sum()
        total_filled += filled_count
        print(f"Filled {filled_count} missing values in '{target_column}' where condition '{condition}' is True with value {fill_value}")
    
    print(f"Total filled: {total_filled} missing values in '{target_column}'")
    return df_copy

def fillna_with_conditional_aggregation(df: pd.DataFrame, target_column: str, 
                                      condition_column: str, condition_values: List[Any],
                                      agg_func: str = 'mean') -> pd.DataFrame:
    """
    Fill missing values using conditional aggregation based on another column's values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_column : str
        Column with missing values to fill
    condition_column : str
        Column to use for conditional grouping
    condition_values : List[Any]
        Values in condition_column to consider for aggregation
    agg_func : str
        Aggregation function ('mean', 'median', 'mode', 'min', 'max')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values filled using conditional aggregation
    """
    df_copy = df.copy()
    
    # Filter data for condition values and calculate statistics
    filtered_data = df_copy[df_copy[condition_column].isin(condition_values)]
    
    if agg_func == 'mode':
        agg_value = filtered_data[target_column].mode()[0] if len(filtered_data[target_column].mode()) > 0 else filtered_data[target_column].mode()
    else:
        agg_value = getattr(filtered_data[target_column], agg_func)()
    
    # Fill missing values where condition_column has specified values
    condition_mask = df_copy[condition_column].isin(condition_values)
    null_mask = df_copy[target_column].isnull()
    combined_mask = condition_mask & null_mask
    
    df_copy.loc[combined_mask, target_column] = agg_value
    
    filled_count = combined_mask.sum()
    print(f"Filled {filled_count} missing values in '{target_column}' where '{condition_column}' is in {condition_values} using {agg_func}: {agg_value}")
    
    return df_copy

def fillna_with_custom_function(df: pd.DataFrame, target_column: str, 
                               condition: str, custom_func: Callable) -> pd.DataFrame:
    """
    Fill missing values using a custom function based on a condition.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_column : str
        Column with missing values to fill
    condition : str
        Pandas condition expression
    custom_func : Callable
        Custom function that takes the filtered DataFrame and returns fill values
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values filled using custom function
    """
    df_copy = df.copy()
    
    # Create mask for rows where condition is True AND target column is null
    condition_mask = df_copy.eval(condition)
    null_mask = df_copy[target_column].isnull()
    combined_mask = condition_mask & null_mask
    
    if combined_mask.sum() > 0:
        # Get the filtered data where condition is True and target is null
        filtered_data = df_copy[combined_mask]
        
        # Apply custom function to get fill values
        fill_values = custom_func(filtered_data)
        
        # Fill the missing values
        df_copy.loc[combined_mask, target_column] = fill_values
        
        filled_count = combined_mask.sum()
        print(f"Filled {filled_count} missing values in '{target_column}' where condition '{condition}' is True using custom function")
    
    return df_copy

def drop_rows_with_missing(df: pd.DataFrame, columns: Union[str, List[str]] = None, 
                          threshold: float = None) -> pd.DataFrame:
    """
    Drop rows with missing values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : str or List[str], optional
        Column(s) to check for missing values. If None, checks all columns.
    threshold : float, optional
        Minimum number of non-null values required to keep a row
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with rows dropped
    """
    df_copy = df.copy()
    original_shape = df_copy.shape
    
    if threshold is not None:
        df_copy = df_copy.dropna(thresh=threshold)
        print(f"Dropped rows with fewer than {threshold} non-null values")
    elif columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        df_copy = df_copy.dropna(subset=columns)
        print(f"Dropped rows with missing values in columns: {columns}")
    else:
        df_copy = df_copy.dropna()
        print("Dropped all rows with any missing values")
    
    dropped_rows = original_shape[0] - df_copy.shape[0]
    print(f"Dropped {dropped_rows} rows. Shape changed from {original_shape} to {df_copy.shape}")
    
    return df_copy

# def dropna(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
#     """
#     Simple dropna wrapper.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     **kwargs : dict
#         Arguments passed to pandas dropna
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with missing values dropped
#     """
#     return df.dropna(**kwargs)

# def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Drop duplicate rows from the DataFrame.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with duplicate rows dropped
#     """
#     df_copy = df.copy()
#     original_shape = df_copy.shape
#     df_copy = df_copy.drop_duplicates()
#     dropped_rows = original_shape[0] - df_copy.shape[0]
#     print(f"Dropped {dropped_rows} duplicate rows. Shape changed from {original_shape} to {df_copy.shape}")
#     return df_copy

# def create_missing_indicator(df: pd.DataFrame, columns: Union[str, List[str]] = None, 
#                            suffix: str = '_missing') -> pd.DataFrame:
#     """
#     Create binary indicators for missing values.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     columns : str or List[str], optional
#         Column(s) to create indicators for. If None, creates for all columns.
#     suffix : str
#         Suffix to add to column names for missing indicators
    
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame with missing indicators added
#     """
#     df_copy = df.copy()
    
#     if columns is None:
#         columns = list(df_copy.columns)
#     elif isinstance(columns, str):
#         columns = [columns]
    
#     for col in columns:
#         if col in df_copy.columns:
#             indicator_name = f"{col}{suffix}"
#             df_copy[indicator_name] = df_copy[col].isnull().astype(int)
#             print(f"Created missing indicator '{indicator_name}' for '{col}'")
    
#     return df_copy

def get_missing_summary(df: pd.DataFrame) -> str:
    """
    Get a summary of missing values in the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns:
    --------
    str : A string representation of the dataframe containing
        Summary of missing values
    """
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    summary = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing_Count': missing_counts.values,
        'Missing_Percentage': missing_percentages.values,
        'Data_Type': df.dtypes.values
    })
    
    summary = summary[summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    print("Missing Value Summary:")
    print(summary.to_string(index=False))
    
    return summary.to_string(index=False)

def cast_columns(df: pd.DataFrame, column_type_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Cast columns to specified data types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column_type_mapping : Dict[str, str]
        Dictionary mapping column names to target data types
        Supported types: 'int', 'float', 'str', 'bool', 'category', 'datetime'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns cast to specified types
    """
    df_copy = df.copy()
    
    for column, target_type in column_type_mapping.items():
        if column in df_copy.columns:
            try:
                if target_type == 'int':
                    df_copy[column] = df_copy[column].astype(int)
                elif target_type == 'float':
                    df_copy[column] = df_copy[column].astype(float)
                elif target_type == 'str':
                    df_copy[column] = df_copy[column].astype(str)
                elif target_type == 'bool':
                    df_copy[column] = df_copy[column].astype(bool)
                elif target_type == 'category':
                    df_copy[column] = df_copy[column].astype('category')
                elif target_type == 'datetime':
                    df_copy[column] = pd.to_datetime(df_copy[column])
                else:
                    print(f"Warning: Unsupported data type '{target_type}' for column '{column}'")
                    continue
                
                print(f"Casted column '{column}' to {target_type}")
            except Exception as e:
                print(f"Error casting column '{column}' to {target_type}: {e}")
    
    return df_copy

def cast_numeric_columns(df: pd.DataFrame, columns: Union[str, List[str]] = None, 
                        target_type: str = 'float') -> pd.DataFrame:
    """
    Cast numeric columns to specified type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : str or List[str], optional
        Column(s) to cast. If None, casts all numeric columns.
    target_type : str
        Target type ('int' or 'float')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with numeric columns cast to specified type
    """
    df_copy = df.copy()
    
    if columns is None:
        # Cast all numeric columns
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_columns)
    elif isinstance(columns, str):
        columns = [columns]
    
    column_type_mapping = {col: target_type for col in columns}
    return cast_columns(df_copy, column_type_mapping)

def cast_integer_columns_to_float(df: pd.DataFrame, columns: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Cast integer columns to float type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : str or List[str], optional
        Column(s) to cast. If None, casts all integer columns.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with integer columns cast to float type
    """
    df_copy = df.copy()
    
    if columns is None:
        # Cast all integer columns
        int_columns = df_copy.select_dtypes(include=['int']).columns
        columns = list(int_columns)
    elif isinstance(columns, str):
        columns = [columns]
    
    column_type_mapping = {col: 'float' for col in columns}
    return cast_columns(df_copy, column_type_mapping)

def cast_categorical_columns(df: pd.DataFrame, columns: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Cast categorical columns to category type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : str or List[str], optional
        Column(s) to cast. If None, casts all object columns.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with categorical columns cast to category type
    """
    df_copy = df.copy()
    
    if columns is None:
        # Cast all object columns
        object_columns = df_copy.select_dtypes(include=['object']).columns
        columns = list(object_columns)
    elif isinstance(columns, str):
        columns = [columns]
    
    column_type_mapping = {col: 'category' for col in columns}
    return cast_columns(df_copy, column_type_mapping)

def one_hot_encode(df: pd.DataFrame, columns: Union[str, List[str]] = None, 
                  drop_first: bool = True, prefix: str = None) -> pd.DataFrame:
    """
    One-hot encode categorical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : str or List[str], optional
        Column(s) to encode. If None, encodes all object/category columns.
    drop_first : bool
        Whether to drop the first category to avoid multicollinearity
    prefix : str, optional
        Prefix for new column names. If None, uses original column name
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with one-hot encoded columns
    """
    df_copy = df.copy()
    
    if columns is None:
        # Encode all categorical columns
        categorical_columns = df_copy.select_dtypes(include=['object', 'category']).columns
        columns = list(categorical_columns)
    elif isinstance(columns, str):
        columns = [columns]
    
    # Filter columns that exist in DataFrame
    columns = [col for col in columns if col in df_copy.columns]
    
    if not columns:
        print("No categorical columns found to encode")
        return df_copy
    
    # Perform one-hot encoding
    encoded_df = pd.get_dummies(df_copy, columns=columns, drop_first=drop_first, prefix=prefix, dtype=int)
    
    new_columns = [col for col in encoded_df.columns if col not in df_copy.columns]
    print(f"One-hot encoded {len(columns)} columns: {columns}")
    print(f"Created {len(new_columns)} new features")
    
    return encoded_df

def label_encode(df: pd.DataFrame, columns: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Label encode categorical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : str or List[str], optional
        Column(s) to encode. If None, encodes all object/category columns.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with label encoded columns
    """
    try:
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("scikit-learn is required for label encoding. Please install it first.")
        return df.copy()
    
    df_copy = df.copy()
    
    if columns is None:
        # Encode all categorical columns
        categorical_columns = df_copy.select_dtypes(include=['object', 'category']).columns
        columns = list(categorical_columns)
    elif isinstance(columns, str):
        columns = [columns]
    
    # Filter columns that exist in DataFrame
    columns = [col for col in columns if col in df_copy.columns]
    
    if not columns:
        print("No categorical columns found to encode")
        return df_copy
    
    # Perform label encoding
    for column in columns:
        le = LabelEncoder()
        df_copy[column] = le.fit_transform(df_copy[column].astype(str))
        print(f"Label encoded column '{column}' with {len(le.classes_)} unique values")
    
    return df_copy

def normalize_features(df: pd.DataFrame, columns: Union[str, List[str]] = None, 
                      method: str = 'standard') -> pd.DataFrame:
    """
    Normalize numeric features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : str or List[str], optional
        Column(s) to normalize. If None, normalizes all numeric columns.
    method : str
        Normalization method ('standard', 'minmax', 'robust')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with normalized features
    """
    df_copy = df.copy()
    
    if columns is None:
        # Normalize all numeric columns
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_columns)
    elif isinstance(columns, str):
        columns = [columns]
    
    # Filter columns that exist in DataFrame
    columns = [col for col in columns if col in df_copy.columns]
    
    if not columns:
        print("No numeric columns found to normalize")
        return df_copy
    
    # Select only numeric columns for normalization
    numeric_df = df_copy[columns].select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        print("No numeric columns found to normalize")
        return df_copy
    
    # Apply normalization
    if method == 'standard':
        scaler = StandardScaler()
        normalized_values = scaler.fit_transform(numeric_df)
        print(f"Standard normalized {len(numeric_df.columns)} columns")
    elif method == 'minmax':
        scaler = MinMaxScaler()
        normalized_values = scaler.fit_transform(numeric_df)
        print(f"Min-max normalized {len(numeric_df.columns)} columns")
    elif method == 'robust':
        scaler = RobustScaler()
        normalized_values = scaler.fit_transform(numeric_df)
        print(f"Robust normalized {len(numeric_df.columns)} columns")
    else:
        print(f"Unknown normalization method: {method}. Using standard normalization.")
        scaler = StandardScaler()
        normalized_values = scaler.fit_transform(numeric_df)
    
    # Update DataFrame with normalized values
    for i, col in enumerate(numeric_df.columns):
        df_copy[col] = normalized_values[:, i]
    
    return df_copy

def encode_all_categorical_columns(df: pd.DataFrame, method: str = 'one_hot', 
                          drop_first: bool = True) -> pd.DataFrame:
    """
    Encode all categorical/object columns using specified method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    method : str
        Encoding method ('one_hot' or 'label')
    drop_first : bool
        Whether to drop first category in one-hot encoding (only applies to one_hot)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with encoded categorical columns
    """
    df_copy = df.copy()
    
    # Get all categorical columns
    categorical_columns = df_copy.select_dtypes(include=['object', 'category']).columns
    categorical_columns = [col for col in categorical_columns if col not in ['_is_train', '_is_test']]
    
    if len(categorical_columns) == 0:
        print("No categorical columns found to encode")
        return df_copy
    
    if method == 'one_hot':
        df_copy = one_hot_encode(df_copy, columns=categorical_columns, drop_first=drop_first)
        print(f"One-hot encoded all {len(categorical_columns)} categorical columns")
    elif method == 'label':
        df_copy = label_encode(df_copy, columns=categorical_columns)
        print(f"Label encoded all {len(categorical_columns)} categorical columns")
    else:
        raise ValueError(f"Unknown encoding method: {method}. Use 'one_hot' or 'label'")
    
    return df_copy

def normalize_all_numerical_columns(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """
    Normalize all numerical columns using specified method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    method : str
        Normalization method ('standard', 'minmax', 'robust')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with normalized numerical columns
    """
    df_copy = df.copy()
    
    # Get all numerical columns
    numerical_columns = df_copy.select_dtypes(include=[np.number]).columns
    numerical_columns = [col for col in numerical_columns if col not in ['_is_train', '_is_test']]
    
    if len(numerical_columns) == 0:
        print("No numerical columns found to normalize")
        return df_copy
    
    df_copy = normalize_features(df_copy, columns=numerical_columns, method=method)
    print(f"Normalized all {len(numerical_columns)} numerical columns using {method} method")
    
    return df_copy

def concatenate_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate train and test data with tracking columns for proper splitting.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training DataFrame
    test_df : pd.DataFrame
        Test DataFrame
    
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with tracking columns '_is_train' and '_is_test'
        
    Notes:
    ------
    The '_is_train' and '_is_test' columns are boolean tracking columns that identify
    which rows originally belonged to train vs test data. This allows proper splitting
    even after transformations that remove rows (like drop_duplicate_rows).
    """


    train_copy = train_df.copy()
    test_copy = test_df.copy()
    train_copy['_is_train'] = True
    train_copy['_is_test'] = False
    test_copy['_is_train'] = False
    test_copy['_is_test'] = True
    
    combined = pd.concat([train_copy, test_copy], ignore_index=True)
    # combined[combined['_is_test'] == True]['']
    print(f"Concatenated train ({train_df.shape}) and test ({test_df.shape}) data")
    print(f"Combined shape: {combined.shape}")
    return combined

def split_combined_into_train_test(combined: pd.DataFrame) -> tuple:
    """
    Split combined data back into train and test using tracking columns.
    
    Parameters:
    -----------
    combined : pd.DataFrame
        Combined DataFrame with tracking columns
    
    Returns:
    --------
    tuple
        (train_df, test_df)
    """
    train_df = combined[combined['_is_train'] == True].drop(['_is_train', '_is_test'], axis=1, errors='ignore')
    test_df = combined[combined['_is_test'] == True].drop(['_is_train', '_is_test'], axis=1, errors='ignore')
    print(f"Split back: train {train_df.shape}, test {test_df.shape}")
    return train_df, test_df

def convert_dataframe_to_features_target(df: pd.DataFrame, target_column: str, is_train: bool = True) -> tuple:
    """
    Convert DataFrame to features and target format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_column : str
        Name of the target column
    is_train : bool
        Whether the DataFrame is the training data
        Set this to True if the DataFrame is the training data, False if it is the test data
    
    Returns:
    --------
    tuple
        (features_df, target_df) if is_train is True, else (features_df, None)
    """
    df_copy = df.copy()
    features_df = df_copy.drop(target_column, axis=1)
    target_df = df_copy[target_column]
    if(is_train):
        return features_df, target_df
    else:
        return features_df, None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def convert_to_dataframe(data: Any, **kwargs) -> pd.DataFrame:
    """
    Convert various data types to pandas DataFrame.
    
    Parameters:
    -----------
    data : Any
        Data to convert. Supported types:
        - pd.Series: Convert to DataFrame with series name as column
        - dict: Convert dictionary to DataFrame
        - list: Convert list of values to DataFrame
        - list of dicts: Convert list of dictionaries to DataFrame
        - list of lists: Convert list of lists to DataFrame
        - numpy array: Convert numpy array to DataFrame
        - scalar: Convert single value to DataFrame
    **kwargs : dict
        Additional arguments passed to pd.DataFrame constructor
    
    Returns:
    --------
    pd.DataFrame
        Converted DataFrame
    
    Examples:
    ---------
    >>> # Convert Series
    >>> series = pd.Series([1, 2, 3], name='values')
    >>> df = convert_to_dataframe(series)
    
    >>> # Convert dictionary
    >>> data_dict = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}
    >>> df = convert_to_dataframe(data_dict)
    
    >>> # Convert list
    >>> data_list = [1, 2, 3, 4, 5]
    >>> df = convert_to_dataframe(data_list, columns=['values'])
    
    >>> # Convert list of dictionaries
    >>> data_list_dict = [{'name': 'John', 'age': 25}, {'name': 'Jane', 'age': 30}]
    >>> df = convert_to_dataframe(data_list_dict)
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    
    elif isinstance(data, pd.Series):
        # Convert Series to DataFrame
        df = data.to_frame()
        print(f"Converted Series '{data.name}' to DataFrame with shape {df.shape}")
        return df
    
    elif isinstance(data, dict):
        # Convert dictionary to DataFrame
        df = pd.DataFrame(data, **kwargs)
        print(f"Converted dictionary to DataFrame with shape {df.shape}")
        return df
    
    elif isinstance(data, list):
        if not data:
            # Empty list
            df = pd.DataFrame(**kwargs)
            print("Converted empty list to empty DataFrame")
            return df
        
        elif isinstance(data[0], dict):
            # List of dictionaries
            df = pd.DataFrame(data, **kwargs)
            print(f"Converted list of {len(data)} dictionaries to DataFrame with shape {df.shape}")
            return df
        
        elif isinstance(data[0], (list, tuple)):
            # List of lists/tuples
            df = pd.DataFrame(data, **kwargs)
            print(f"Converted list of {len(data)} lists to DataFrame with shape {df.shape}")
            return df
        
        else:
            # Simple list of values
            df = pd.DataFrame(data, **kwargs)
            print(f"Converted list of {len(data)} values to DataFrame with shape {df.shape}")
            return df
    
    elif hasattr(data, '__array__'):
        # Numpy array or similar
        df = pd.DataFrame(data, **kwargs)
        print(f"Converted array-like object to DataFrame with shape {df.shape}")
        return df
    
    else:
        # Scalar or other types
        try:
            df = pd.DataFrame([data], **kwargs)
            print(f"Converted scalar/other type to DataFrame with shape {df.shape}")
            return df
        except Exception as e:
            print(f"Error converting data to DataFrame: {e}")
            return pd.DataFrame()

def drop_feature(df: pd.DataFrame, column: Union[str, List[str]]) -> pd.DataFrame:
    """
    Drop feature(s) from the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column : str or List[str]
        Name(s) of the column(s) to drop
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with column(s) removed
    """
    df_copy = df.copy()
    if isinstance(column, str):
        column = [column]
    
    columns_to_drop = [col for col in column if col in df_copy.columns]
    if columns_to_drop:
        df_copy = df_copy.drop(columns=columns_to_drop)
        print(f"Dropped features: {columns_to_drop}")
    return df_copy

def get_features(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Extract specific features (columns) from the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : str or List[str]
        Name(s) of the column(s) to extract
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with only the specified columns
    """
    df_copy = df.copy()
    if isinstance(columns, str):
        columns = [columns]
    
    # Filter columns that exist in DataFrame
    existing_columns = [col for col in columns if col in df_copy.columns]
    missing_columns = [col for col in columns if col not in df_copy.columns]
    
    if missing_columns:
        raise ValueError(f"Warning: Columns not found: {missing_columns}")
    
    if existing_columns:
        df_copy = df_copy[existing_columns]
        print(f"Extracted features: {existing_columns}")
    else:
        raise ValueError("No valid columns found to extract")
    
    return df_copy

def concatenate_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    """
    Concatenate two DataFrames.
    
    Parameters:
    -----------
    df1 : pd.DataFrame
        First DataFrame to concatenate
    df2 : pd.DataFrame
        Second DataFrame to concatenate
    axis : int
        Axis to concatenate along (0 for rows, 1 for columns)
    
    Returns:
    --------
    pd.DataFrame
        Concatenated DataFrame
    """
    # Check if DataFrames are empty
    if df1.empty and df2.empty:
        print("Both DataFrames are empty")
        return pd.DataFrame()
    
    if df1.empty:
        print("First DataFrame is empty, returning second DataFrame")
        return df2.copy()
    
    if df2.empty:
        print("Second DataFrame is empty, returning first DataFrame")
        return df1.copy()
    
    # Concatenate DataFrames
    result = pd.concat([df1, df2], axis=axis)
    
    print(f"Concatenated 2 DataFrames along axis {axis}")
    print(f"First DataFrame shape: {df1.shape}")
    print(f"Second DataFrame shape: {df2.shape}")
    print(f"Result shape: {result.shape}")
    
    return result

def join_dataframes(left_df: pd.DataFrame, right_df: pd.DataFrame, 
                   left_on: Union[str, List[str]], right_on: Union[str, List[str]] = None,
                   how: str = 'inner', suffixes: tuple = ('_x', '_y')) -> pd.DataFrame:
    """
    Join two DataFrames using pandas merge functionality.
    
    Parameters:
    -----------
    left_df : pd.DataFrame
        Left DataFrame to join
    right_df : pd.DataFrame
        Right DataFrame to join
    left_on : str or List[str]
        Column(s) from left DataFrame to use as join key(s)
    right_on : str or List[str], optional
        Column(s) from right DataFrame to use as join key(s). If None, uses same as left_on
    how : str
        Type of join ('inner', 'left', 'right', 'outer')
    suffixes : tuple
        Suffixes to append to overlapping column names
    
    Returns:
    --------
    pd.DataFrame
        Joined DataFrame
    """
    if right_on is None:
        right_on = left_on
    
    # Validate join keys exist in DataFrames
    if isinstance(left_on, str):
        left_on = [left_on]
    if isinstance(right_on, str):
        right_on = [right_on]
    
    missing_left = [col for col in left_on if col not in left_df.columns]
    missing_right = [col for col in right_on if col not in right_df.columns]
    
    if missing_left:
        raise ValueError(f"Join keys not found in left DataFrame: {missing_left}")
    
    if missing_right:
        raise ValueError(f"Join keys not found in right DataFrame: {missing_right}")
    
    # Perform the join
    result = pd.merge(left_df, right_df, left_on=left_on, right_on=right_on, 
                     how=how, suffixes=suffixes)
    
    print(f"Joined DataFrames using {how} join")
    print(f"Left DataFrame shape: {left_df.shape}")
    print(f"Right DataFrame shape: {right_df.shape}")
    print(f"Result shape: {result.shape}")
    print(f"Join keys: left={left_on}, right={right_on}")
    
    return result

def rename_feature(df: pd.DataFrame, old_name: Union[str, List[str]], new_name: Union[str, List[str]]) -> pd.DataFrame:
    """
    Rename feature(s).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    old_name : str or List[str]
        Current name(s) of the column(s)
    new_name : str or List[str]
        New name(s) for the column(s)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with renamed column(s)
    """
    df_copy = df.copy()
    
    if isinstance(old_name, str):
        old_name = [old_name]
    if isinstance(new_name, str):
        new_name = [new_name]
    
    if len(old_name) != len(new_name):
        print("Error: Number of old names must match number of new names")
        return df_copy
    
    rename_mapping = {}
    for old, new in zip(old_name, new_name):
        if old in df_copy.columns:
            rename_mapping[old] = new
    
    if rename_mapping:
        df_copy = df_copy.rename(columns=rename_mapping)
        print(f"Renamed features: {list(rename_mapping.items())}")
    
    return df_copy

# def show_feature_info(df: pd.DataFrame, original_columns: List[str] = None) -> str:
#     """
#     Display information about all features.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     original_columns : List[str], optional
#         List of original column names to compare against
        
#     Returns:
#     --------
#     str
#         Formatted string containing feature information
#     """
#     if original_columns is None:
#         original_columns = []
    
#     new_features = [col for col in df.columns if col not in original_columns]
    
#     result = []
#     result.append(f"\nDataFrame Shape: {df.shape}")
#     result.append(f"Original Features: {len(original_columns)}")
#     result.append(f"New Features: {len(new_features)}")
#     result.append(f"Total Features: {len(df.columns)}")
    
#     result.append(f"\nNew Features Created:")
#     for feature in new_features:
#         dtype = df[feature].dtype
#         null_count = df[feature].isnull().sum()
#         result.append(f"  {feature}: {dtype} (nulls: {null_count})")
    
#     return "\n".join(result)

# ============================================================================
# DATA EXPLORATION AND SUMMARY TOOLS
# ============================================================================

def get_unique_values(df: pd.DataFrame, column: str, sort: bool = True, 
                     include_counts: bool = True) -> pd.DataFrame:
    """
    Get unique values from a column as a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column name to get unique values from
    sort : bool
        Whether to sort the unique values
    include_counts : bool
        Whether to include value counts
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with unique values and optionally their counts
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame")
        return pd.DataFrame()
    
    if include_counts:
        value_counts = df[column].value_counts()
        if not sort:
            value_counts = value_counts.sort_index()
        
        result_df = pd.DataFrame({
            'value': value_counts.index,
            'count': value_counts.values,
            'percentage': (value_counts.values / len(df)) * 100
        })
    else:
        unique_vals = df[column].unique()
        if sort:
            unique_vals = sorted(unique_vals)
        
        result_df = pd.DataFrame({
            'value': unique_vals
        })
    
    return result_df

# def get_column_summary(df: pd.DataFrame, column: str) -> str:
#     """
#     Get comprehensive summary of a single column as a formatted string.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     column : str
#         Column name to summarize
    
#     Returns:
#     --------
#     str
#         Formatted string containing column summary statistics
#     """
#     if column not in df.columns:
#         return f"Column '{column}' not found in DataFrame"
    
#     col_data = df[column]
#     dtype = col_data.dtype
    
#     result = []
#     result.append(f"=== Column Summary: {column} ===")
#     result.append(f"Data Type: {dtype}")
#     result.append(f"Total Count: {len(col_data)}")
#     result.append(f"Non-null Count: {col_data.count()}")
#     result.append(f"Null Count: {col_data.isnull().sum()} ({(col_data.isnull().sum() / len(col_data)) * 100:.2f}%)")
#     result.append(f"Unique Values: {col_data.nunique()}")
    
#     # Numeric column statistics
#     if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
#         result.append(f"\nNumeric Statistics:")
#         result.append(f"  Mean: {col_data.mean():.4f}")
#         result.append(f"  Median: {col_data.median():.4f}")
#         result.append(f"  Std: {col_data.std():.4f}")
#         result.append(f"  Min: {col_data.min()}")
#         result.append(f"  Max: {col_data.max()}")
#         result.append(f"  Q25: {col_data.quantile(0.25):.4f}")
#         result.append(f"  Q75: {col_data.quantile(0.75):.4f}")
    
#     # Categorical/object column statistics
#     if pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
#         value_counts = col_data.value_counts()
#         if len(value_counts) > 0:
#             result.append(f"\nCategorical Statistics:")
#             result.append(f"  Most Common: '{value_counts.index[0]}' ({value_counts.iloc[0]} times)")
#             result.append(f"  Least Common: '{value_counts.index[-1]}' ({value_counts.iloc[-1]} times)")
    
#     # Datetime column statistics
#     if pd.api.types.is_datetime64_any_dtype(col_data):
#         if col_data.count() > 0:
#             result.append(f"\nDatetime Statistics:")
#             result.append(f"  Year Range: {col_data.dt.year.min()}-{col_data.dt.year.max()}")
#             if len(col_data.dt.year.mode()) > 0:
#                 result.append(f"  Most Common Year: {col_data.dt.year.mode()[0]}")
#             if len(col_data.dt.month.mode()) > 0:
#                 result.append(f"  Most Common Month: {col_data.dt.month.mode()[0]}")
#             if len(col_data.dt.dayofweek.mode()) > 0:
#                 result.append(f"  Most Common Day of Week: {col_data.dt.dayofweek.mode()[0]}")
#             result.append(f"  Weekend Percentage: {(col_data.dt.dayofweek >= 5).sum() / col_data.count() * 100:.1f}%")
    
#     return "\n".join(result)

# def print_column_summary(df: pd.DataFrame, column: str) -> str:
#     """
#     Print a formatted summary of a column.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     column : str
#         Column name to summarize
    
#     Returns:
#     --------
#     str
#         Formatted string containing column summary
#     """
#     summary = get_column_summary(df, column)
#     print(summary)
#     return summary

def get_dataframe_dtypes_summary(df: pd.DataFrame) -> str:
    """
    Get comprehensive summary of the dtypes in the entire DataFrame as a formatted string.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns:
    --------
    str
        Formatted string containing dtypes summary of all columns in the dataframe
    """
    result = []
    result.append(f"=== DATAFRAME SUMMARY ===")
    result.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    result.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types summary
    dtype_counts = df.dtypes.value_counts()
    result.append(f"\nData Types:")
    for dtype, count in dtype_counts.items():
        result.append(f"  {dtype}: {count} columns")
    return "\n".join(result)

# def get_dataframe_summary(df: pd.DataFrame) -> str:
#     """
#     Get comprehensive summary of the entire DataFrame as a formatted string.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
    
#     Returns:
#     --------
#     str
#         Formatted string containing summary of all columns
#     """
#     result = []
#     result.append(f"=== DATAFRAME SUMMARY ===")
#     result.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
#     result.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
#     # Data types summary
#     dtype_counts = df.dtypes.value_counts()
#     result.append(f"\nData Types:")
#     for dtype, count in dtype_counts.items():
#         result.append(f"  {dtype}: {count} columns")
    
#     # Missing values summary
#     missing_summary = get_missing_summary(df)
#     if missing_summary:
#         result.append(f"\nColumns with Missing Values:")
#         result.append(missing_summary)
#     else:
#         result.append(f"\nNo missing values found in any column.")
    
#     # Detailed column summaries
#     result.append(f"\nDetailed Column Summaries:")
#     for column in df.columns:
#         col_summary = get_column_summary(df, column)
#         result.append(col_summary)
#         result.append("")  # Add spacing between columns
    
#     return "\n".join(result)

# def print_dataframe_summary(df: pd.DataFrame) -> str:
#     """
#     Print a formatted summary of the entire DataFrame.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
        
#     Returns:
#     --------
#     str
#         Formatted string containing DataFrame summary
#     """
#     summary = get_dataframe_summary(df)
#     print(summary)
#     return summary

# def explore_categorical_column(df: pd.DataFrame, column: str, top_n: int = 10) -> str:
#     """
#     Explore a categorical column in detail.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     column : str
#         Categorical column to explore
#     top_n : int
#         Number of top values to show
        
#     Returns:
#     --------
#     str
#         Formatted string containing categorical column exploration
#     """
#     if column not in df.columns:
#         error_msg = f"Column '{column}' not found in DataFrame"
#         print(error_msg)
#         return error_msg
    
#     result = []
#     result.append(f"=== EXPLORING CATEGORICAL COLUMN: {column} ===")
    
#     # Basic info
#     col_summary = print_column_summary(df, column)
#     result.append(col_summary)
    
#     # Value counts
#     value_counts = df[column].value_counts()
#     result.append(f"\nValue Counts (Top {top_n}):")
#     for i, (value, count) in enumerate(value_counts.head(top_n).items()):
#         percentage = (count / len(df)) * 100
#         result.append(f"  {i+1:2d}. '{value}': {count:6d} ({percentage:5.1f}%)")
    
#     if len(value_counts) > top_n:
#         remaining = len(value_counts) - top_n
#         remaining_count = value_counts.iloc[top_n:].sum()
#         remaining_percentage = (remaining_count / len(df)) * 100
#         result.append(f"  ... and {remaining} more values: {remaining_count:6d} ({remaining_percentage:5.1f}%)")
    
#     # Missing values
#     null_count = df[column].isnull().sum()
#     if null_count > 0:
#         result.append(f"\nMissing Values: {null_count} ({null_count/len(df)*100:.1f}%)")
    
#     # Sample values
#     result.append(f"\nSample Values:")
#     sample_values = df[column].dropna().unique()[:5]
#     for i, value in enumerate(sample_values):
#         result.append(f"  {i+1}. '{value}'")
    
#     result_str = "\n".join(result)
#     print(result_str)
#     return result_str

# def explore_numeric_column(df: pd.DataFrame, column: str, bins: int = 10) -> str:
#     """
#     Explore a numeric column in detail.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     column : str
#         Numeric column to explore
#     bins : int
#         Number of bins for histogram
        
#     Returns:
#     --------
#     str
#         Formatted string containing numeric column exploration
#     """
#     if column not in df.columns:
#         error_msg = f"Column '{column}' not found in DataFrame"
#         print(error_msg)
#         return error_msg
    
#     if not pd.api.types.is_numeric_dtype(df[column]):
#         error_msg = f"Column '{column}' is not numeric"
#         print(error_msg)
#         return error_msg
    
#     result = []
#     result.append(f"=== EXPLORING NUMERIC COLUMN: {column} ===")
    
#     # Basic info
#     col_summary = print_column_summary(df, column)
#     result.append(col_summary)
    
#     # Distribution info
#     col_data = df[column].dropna()
#     if len(col_data) > 0:
#         result.append(f"\nDistribution Statistics:")
#         result.append(f"  Skewness: {col_data.skew():.4f}")
#         result.append(f"  Kurtosis: {col_data.kurtosis():.4f}")
        
#         # Percentiles
#         percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
#         result.append(f"\nPercentiles:")
#         for p in percentiles:
#             value = col_data.quantile(p/100)
#             result.append(f"  {p:2d}th: {value:.4f}")
        
#         # Histogram bins
#         result.append(f"\nHistogram Bins ({bins} bins):")
#         hist, bin_edges = np.histogram(col_data, bins=bins)
#         for i in range(len(hist)):
#             if hist[i] > 0:
#                 result.append(f"  {bin_edges[i]:8.2f} - {bin_edges[i+1]:8.2f}: {hist[i]:4d} values")
    
#     # Missing values
#     null_count = df[column].isnull().sum()
#     if null_count > 0:
#         result.append(f"\nMissing Values: {null_count} ({null_count/len(df)*100:.1f}%)")
    
#     result_str = "\n".join(result)
#     print(result_str)
#     return result_str

# def compare_columns(df: pd.DataFrame, columns: List[str]) -> str:
#     """
#     Compare multiple columns side by side.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame
#     columns : List[str]
#         List of columns to compare
        
#     Returns:
#     --------
#     str
#         Formatted string containing column comparison
#     """
#     result = []
#     result.append(f"=== COLUMN COMPARISON ===")
    
#     # Check if all columns exist
#     missing_cols = [col for col in columns if col not in df.columns]
#     if missing_cols:
#         result.append(f"Missing columns: {missing_cols}")
#         return "\n".join(result)
    
#     # Create comparison summaries
#     for column in columns:
#         summary = get_column_summary(df, column)
#         result.append(summary)
#         result.append("")  # Add spacing between columns
    
#     return "\n".join(result)

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# def example_usage():
#     """Example usage of the feature engineering tools."""
#     # Create sample data
#     data = {
#         'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
#         'age': [25, 30, 35],
#         'salary': [50000, 60000, 70000],
#         'department': ['IT', 'HR', 'IT'],
#         'join_date': ['2020-01-15', '2019-06-20', '2021-03-10']
#     }
#     df = pd.DataFrame(data)
#     original_columns = list(df.columns)
    
#     # Example feature engineering operations
#     df = extract_string_pattern(df, 'last_name', 'name', r'(\w+)$')
#     df = string_length(df, 'name_length', 'name')
#     df = create_numeric_feature(df, 'age_salary_ratio', 'age / salary')
#     df = create_conditional_feature(df, 'high_salary', 'salary > 55000', 1, 0)
#     df = create_group_aggregation(df, 'dept_avg_salary', 'department', 'salary', 'mean')
#     df = extract_datetime_features(df, 'join_date', ['year', 'month', 'dayofweek'])
#     df = create_interaction_features(df, ['age', 'salary'], 'product')
    
#     feature_info = show_feature_info(df, original_columns)
#     print(feature_info)
    
#     return df

# if __name__ == "__main__":
#     # Run example
#     result_df = example_usage()
#     print("\nFinal DataFrame:")
#     print(result_df) 

def filter_dataframe(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """
    Filter DataFrame using a boolean condition.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    condition : str
        Boolean condition expression (e.g., 'col1 > 0', 'col1 == "value"', 'col1 > 0 and col2 < 100')
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame containing only rows that satisfy the condition
        
    Examples:
    ---------
    >>> # Filter rows where age > 25
    >>> filtered_df = filter_dataframe(df, 'age > 25')
    
    >>> # Filter rows where salary > 50000 and department == "IT"
    >>> filtered_df = filter_dataframe(df, 'salary > 50000 and department == "IT"')
    
    >>> # Filter rows where column is not null
    >>> filtered_df = filter_dataframe(df, 'col1.notna()')
    """
    try:
        # Use pandas eval to evaluate the condition
        mask = df.eval(condition)
        filtered_df = df[mask].copy()
        
        original_count = len(df)
        filtered_count = len(filtered_df)
        removed_count = original_count - filtered_count
        
        print(f"Filtered DataFrame using condition: {condition}")
        print(f"Original rows: {original_count}")
        print(f"Filtered rows: {filtered_count}")
        print(f"Removed rows: {removed_count} ({removed_count/original_count*100:.1f}%)")
        
        return filtered_df
        
    except Exception as e:
        raise ValueError(f"Error evaluating condition '{condition}': {e}")

# def observe_dataframe(df: pd.DataFrame, max_rows: int = 30) -> str:
#     """
#     Return a string representation of the dataframe to look at.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input DataFrame to print
#     max_rows : int
#         Maximum number of rows to display (default: 30)
    
#     Returns:
#     --------
#     str
#         String representation of the DataFrame
#     """
#     if df.empty:
#         return "DataFrame is empty"
    
#     if len(df) > max_rows:
#         return df.head(max_rows).to_string(index=False)
#     else:
#         return df.to_string(index=False)