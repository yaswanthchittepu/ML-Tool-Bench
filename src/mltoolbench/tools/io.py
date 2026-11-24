"""Reading data and persisting models / dataframes."""

import pickle
from typing import Any

import pandas as pd


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


def save_model(model: Any, filepath: str = 'model.pkl') -> str:
    """
    Save the trained model to disk using pickle.
    
    Parameters
    ----------
    model : Any
        Trained model to save
    filepath : str
        Path where to save the model file (e.g., 'model.pkl')
        
    Returns
    -------
    str
        Confirmation message with the save location.
        
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    return f"Model saved successfully to {filepath}"


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk using pickle.
    
    Parameters
    ----------
    filepath : str
        Path to the saved model file (e.g., 'model.pkl')
        
    Returns
    -------
    Any
        Loaded model
        
    Raises
    ------
    FileNotFoundError
        If model file is not found
    Exception
        If there's an error loading the model
    """
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {filepath}")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")


def save_dataframe_to_csv(df: pd.DataFrame, filepath: str) -> str:
    """
    Save a DataFrame to CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str
        Path where to save the CSV file (e.g., 'data.csv')
        
    Returns
    -------
    str
        Confirmation message with the save location
    """
    df.to_csv(filepath, index=False)
    return f"DataFrame saved successfully to {filepath}"
