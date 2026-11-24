"""Missing-value handling, type casting and data-quality inspection."""

from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd


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
