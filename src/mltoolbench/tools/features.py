"""Feature construction, encoding and scaling."""

from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


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
