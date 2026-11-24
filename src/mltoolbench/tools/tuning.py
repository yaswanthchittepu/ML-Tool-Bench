"""Hyperparameter tuning via cross-validated grid search.

Every ``tune_*`` tool returns ``(best_estimator, best_params, best_cv_score)``.
"""


from typing import Any, Tuple

import catboost as cb
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV

MODEL_JOBS = 1
GRID_SEARCH_JOBS = 1


def _grid_search(model: Any, param_grid: dict, X_train: pd.DataFrame, y_train: pd.Series,
                 cv: int, scoring: str, label: str) -> Tuple[Any, dict, float]:
    try:
        print(f"Started tuning {label}")
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=GRID_SEARCH_JOBS)
        grid_search.fit(X_train, y_train)
        print(f"Finished tuning {label}")
        print(f"Best parameters: {grid_search.best_params_}, Best CV score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    except Exception as e:
        print(f"Error tuning {label}: {e}")
        raise


def tune_logistic_regressor(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Tuple[LogisticRegression, dict, float]:
    """
    Perform hyperparameter tuning for Logistic Regression using GridSearchCV.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    cv : int, default=5
        Number of cross-validation folds.
        
    Returns
    -------
    Tuple[LogisticRegression, dict, float]
        Best tuned logistic regression model, best parameters, and best CV score
        
    """
    param_grid = {
        'C': [0.001, 0.01, 0.1],
        'penalty': ['l2'],
    }
    model = LogisticRegression(random_state=42, n_jobs=MODEL_JOBS)
    return _grid_search(model, param_grid, X_train, y_train, cv=cv, scoring='accuracy', label='logistic regressor')


def tune_linear_regressor(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Tuple[LinearRegression, dict, float]:
    """
    Perform hyperparameter tuning for Linear Regression using GridSearchCV.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    cv : int, default=5
        Number of cross-validation folds.
        
    Returns
    -------
    Tuple[LinearRegression, dict, float]
        Best tuned linear regression model, best parameters, and best CV score
        
    """
    param_grid = {
        'fit_intercept': [True],
    }
    model = LinearRegression(n_jobs=MODEL_JOBS)
    return _grid_search(model, param_grid, X_train, y_train, cv=cv, scoring='r2', label='linear regression')


def tune_random_forest_regressor(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Tuple[RandomForestRegressor, dict, float]:
    """
    Perform hyperparameter tuning for Random Forest Regressor using GridSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    cv : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    Tuple[RandomForestRegressor, dict, float]
        Best tuned random forest regressor model, best parameters, and best CV score
    """
    param_grid = {
        'n_estimators': [100],
        'max_depth': [20],
        'min_samples_split': [2],
        'min_samples_leaf': [2],
    }
    model = RandomForestRegressor(random_state=42, n_jobs=MODEL_JOBS)
    return _grid_search(model, param_grid, X_train, y_train, cv=cv, scoring='r2', label='random forest regressor')


def tune_random_forest_classifier(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Tuple[RandomForestClassifier, dict, float]:
    """
    Perform hyperparameter tuning for Random Forest Classifier using GridSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    cv : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    Tuple[RandomForestClassifier, dict, float]
        Best tuned random forest classifier model, best parameters, and best CV score
    """
    param_grid = {
        'n_estimators': [200],
        'max_depth': [10],
        'min_samples_split': [10],
        'min_samples_leaf': [4],
    }
    model = RandomForestClassifier(random_state=42, n_jobs=MODEL_JOBS)
    return _grid_search(model, param_grid, X_train, y_train, cv=cv, scoring='accuracy', label='random forest classifier')


def tune_xgboost_regressor(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Tuple[xgb.XGBRegressor, dict, float]:
    """
    Perform hyperparameter tuning for XGBoost Regressor using GridSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    cv : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    Tuple[xgb.XGBRegressor, dict, float]
        Best tuned XGBoost regressor model, best parameters, and best CV score
    """
    param_grid = {
        'n_estimators': [100],
        'max_depth': [20],
        'subsample': [1.0],
    }
    model = xgb.XGBRegressor(random_state=42, n_jobs=MODEL_JOBS)
    return _grid_search(model, param_grid, X_train, y_train, cv=cv, scoring='r2', label='XGBoost regressor')


def tune_xgboost_classifier(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Tuple[xgb.XGBClassifier, dict, float]:
    """
    Perform hyperparameter tuning for XGBoost Classifier using GridSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    cv : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    Tuple[xgb.XGBClassifier, dict, float]
        Best tuned XGBoost classifier model, best parameters, and best CV score
    """
    param_grid = {
        'n_estimators': [100],
        'max_depth': [20],
        'subsample': [1.0],
    }
    model = xgb.XGBClassifier(random_state=42, n_jobs=MODEL_JOBS)
    return _grid_search(model, param_grid, X_train, y_train, cv=cv, scoring='accuracy', label='XGBoost classifier')


def tune_lightgbm_regressor(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Tuple[lgb.LGBMRegressor, dict, float]:
    """
    Perform hyperparameter tuning for LightGBM Regressor using GridSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    cv : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    Tuple[lgb.LGBMRegressor, dict, float]
        Best tuned LightGBM regressor model, best parameters, and best CV score
    """
    param_grid = {
        'n_estimators': [100],
        'max_depth': [20],
        'learning_rate': [0.1],
        'subsample': [0.8],
    }
    model = lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=MODEL_JOBS)
    return _grid_search(model, param_grid, X_train, y_train, cv=cv, scoring='r2', label='LightGBM regressor')


def tune_lightgbm_classifier(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Tuple[lgb.LGBMClassifier, dict, float]:
    """
    Perform hyperparameter tuning for LightGBM Classifier using GridSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    cv : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    Tuple[lgb.LGBMClassifier, dict, float]
        Best tuned LightGBM classifier model, best parameters, and best CV score
    """
    param_grid = {
        'n_estimators': [100],
        'max_depth': [20],
        'learning_rate': [0.01],
        'subsample': [0.8],
    }
    model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=MODEL_JOBS)
    return _grid_search(model, param_grid, X_train, y_train, cv=cv, scoring='accuracy', label='LightGBM classifier')


def tune_catboost_regressor(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Tuple[cb.CatBoostRegressor, dict, float]:
    """
    Perform hyperparameter tuning for CatBoost Regressor using GridSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    cv : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    Tuple[cb.CatBoostRegressor, dict, float]
        Best tuned CatBoost regressor model, best parameters, and best CV score
    """
    param_grid = {
        'iterations': [50],
        'depth': [6],
        'learning_rate': [0.1],
        'l2_leaf_reg': [1],
    }
    model = cb.CatBoostRegressor(random_state=42, verbose=False, thread_count=MODEL_JOBS)
    return _grid_search(model, param_grid, X_train, y_train, cv=cv, scoring='r2', label='CatBoost regressor')


def tune_catboost_classifier(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Tuple[cb.CatBoostClassifier, dict, float]:
    """
    Perform hyperparameter tuning for CatBoost Classifier using GridSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    cv : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    Tuple[cb.CatBoostClassifier, dict, float]
        Best tuned CatBoost classifier model, best parameters, and best CV score
    """
    param_grid = {
        'iterations': [50],
        'depth': [6],
        'learning_rate': [0.1],
        'l2_leaf_reg': [1],
    }
    model = cb.CatBoostClassifier(random_state=42, verbose=False, thread_count=MODEL_JOBS)
    return _grid_search(model, param_grid, X_train, y_train, cv=cv, scoring='accuracy', label='CatBoost classifier')
