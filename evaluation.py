import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from typing import Any

############## EVALUATING MODELS ##############

def evaluate_regression_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "model", eval_data_label: str = 'test') -> str:
    """
    Evaluate a trained regression model on data.

    This function is used to evaluate the performance of a trained regression model on a given dataset.

    The dataset can be the anything, such as the test set or the train set.

    Use this to evaluate the performance of a trained regression model on some dataset, to make informed choices about what to do next.

    Parameters
    ----------
    model : Any
        Trained regression model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    model_name : str, default="model"
        Name of the model for reporting
    eval_data_label : str, default='test'
        Label of the data to evaluate on.
        Can be 'test' or 'train'
        
    Returns
    -------
    str
        A string with the evaluation results.
        
    """
    print(f"Started evaluating {model_name} on {eval_data_label} data")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Finished evaluating {model_name} on {eval_data_label} data. MSE: {mse:.4f}, R2: {r2:.4f}")
    return f"On the {eval_data_label} data, the model: {model_name} has a MSE of {mse:.4f} and an R2 score of {r2:.4f}"


def evaluate_classification_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "model", eval_data_label: str = 'test') -> str:
    """
    Evaluate a trained classification model on data.

    This function is used to evaluate the performance of a trained classification model on a given dataset.

    The dataset can be the anything, such as the test set or the train set.

    Use this to evaluate the performance of a trained classification model on some dataset, to make informed choices about what to do next.

    Parameters
    ----------
    model : Any
        Trained classification model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    model_name : str, default="model"
        Name of the model for reporting
    eval_data_label : str, default='test'
        Label of the data to evaluate on.
        Can be 'test' or 'train'
        
    Returns
    -------
    str
        A string with the evaluation results.
        
    """
    print(f"Started evaluating {model_name} on {eval_data_label} data")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred)
    print(f"Finished evaluating {model_name} on {eval_data_label} data. Accuracy: {accuracy:.4f}, Classification Report: {cls_report}")
    return f"On the {eval_data_label} data, the model: {model_name} has an accuracy of {accuracy:.4f}. The classification report is: {cls_report}"


def predict_target(model: Any, X_data: pd.DataFrame, model_name: str = "model", return_probabilities: bool = False) -> pd.DataFrame:
    """
    Make predictions using a trained model.
    
    Parameters
    ----------
    model : Any
        Trained model (regression or classification)
    X_data : pd.DataFrame
        Features to make predictions on
    model_name : str, default="model"
        Name of the model for reporting
    return_probabilities : bool, default=False
        If True, return class probabilities instead of class predictions (only for classification models)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with predicted target values or probabilities
    """
    print(f"Making predictions using {model_name} on {X_data.shape[0]} samples")
    
    if return_probabilities:
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X_data)
            print(f"Generated probability predictions with shape {predictions.shape}")
            
            # Create column names for each class probability
            if hasattr(model, 'classes_'):
                columns = [f'prob_class_{cls}' for cls in model.classes_]
            else:
                columns = [f'prob_class_{i}' for i in range(predictions.shape[1])]
            
            return pd.DataFrame(predictions, columns=columns)
        else:
            print(f"Warning: Model {model_name} does not support predict_proba(). Falling back to regular predictions.")
            predictions = model.predict(X_data)
            print(f"Generated {len(predictions)} predictions")
            return pd.DataFrame({'predicted_target': predictions})
    else:
        predictions = model.predict(X_data)
        print(f"Generated {len(predictions)} predictions")
        return pd.DataFrame({'predicted_target': predictions})

