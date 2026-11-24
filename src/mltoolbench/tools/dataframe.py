"""Structural dataframe operations: combine/split, select, join, rename, inspect."""

from typing import Any, List, Union

import pandas as pd


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
    result.append("=== DATAFRAME SUMMARY ===")
    result.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    result.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types summary
    dtype_counts = df.dtypes.value_counts()
    result.append("\nData Types:")
    for dtype, count in dtype_counts.items():
        result.append(f"  {dtype}: {count} columns")
    return "\n".join(result)


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
