import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import GridSearchCV
from typing import Tuple, Any


MODEL_JOBS = 1
GRID_SEARCH_JOBS = 1

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
    
    try:
        print(f"Started tuning logistic regressor")
        model = LogisticRegression(random_state=42, n_jobs=MODEL_JOBS)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy', n_jobs=GRID_SEARCH_JOBS
        )
        grid_search.fit(X_train, y_train)
        print(f"Finished tuning logistic regressor")
        print(f"Best parameters: {grid_search.best_params_}, Best CV score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    except Exception as e:
        print(f"Error tuning logistic regressor: {e}")
        raise e


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
    
    try:
        print(f"Started tuning linear regression")
        model = LinearRegression(n_jobs=MODEL_JOBS)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='r2', n_jobs=GRID_SEARCH_JOBS
        )
        grid_search.fit(X_train, y_train)
        print(f"Finished tuning linear regression")
        print(f"Best parameters: {grid_search.best_params_}, Best CV score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    except Exception as e:
        print(f"Error tuning linear regression: {e}")
        raise e


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
    # param_grid = {
    #     'n_estimators': [50, 100],
    #     'max_depth': [10, 20],
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 2],
    # }

    param_grid = {
        'n_estimators': [100],
        'max_depth': [20],
        'min_samples_split': [2],
        'min_samples_leaf': [2],
    }

    try:
        print(f"Started tuning random forest regressor")
        model = RandomForestRegressor(random_state=42, n_jobs=MODEL_JOBS)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='r2', n_jobs=GRID_SEARCH_JOBS
        )
        grid_search.fit(X_train, y_train)
        print(f"Finished tuning random forest regressor")
        print(f"Best parameters: {grid_search.best_params_}, Best CV score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    except Exception as e:
        print(f"Error tuning random forest regressor: {e}")
        raise e


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
    # param_grid = {
    #     'n_estimators': [100],
    #     'max_depth': [10, 20],
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 2],
    # }

    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [10, 20, None],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['sqrt', 'log2', None],
    # }
    # Best parameters: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 200}, Best CV score: 0.7993

    param_grid = {
        'n_estimators': [200],
        'max_depth': [10],
        'min_samples_split': [10],
        'min_samples_leaf': [4],
    }

    try:
        print(f"Started tuning random forest classifier")
        model = RandomForestClassifier(random_state=42, n_jobs=MODEL_JOBS)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy', n_jobs=GRID_SEARCH_JOBS
        )
        grid_search.fit(X_train, y_train)
        print(f"Finished tuning random forest classifier")
        print(f"Best parameters: {grid_search.best_params_}, Best CV score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    except Exception as e:
        print(f"Error tuning random forest classifier: {e}")
        raise e


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
    # param_grid = {
    #     'n_estimators': [50, 100],
    #     'max_depth': [10, 20],
    #     'subsample': [0.8, 1.0],
    # }

    param_grid = {
        'n_estimators': [100],
        'max_depth': [20],
        'subsample': [1.0],
    }

    print(f"Started tuning XGBoost regressor")
    try:
        model = xgb.XGBRegressor(random_state=42, n_jobs=MODEL_JOBS)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='r2', n_jobs=GRID_SEARCH_JOBS
        )
        grid_search.fit(X_train, y_train)
        print(f"Finished tuning XGBoost regressor")
        print(f"Best parameters: {grid_search.best_params_}, Best CV score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    except Exception as e:
        print(f"Error tuning XGBoost regressor: {e}")
        raise e



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
    # param_grid = {
    #     'n_estimators': [50, 100],
    #     'max_depth': [10, 20],
    #     'subsample': [0.8, 1.0],
    # }

    param_grid = {
        'n_estimators': [100],
        'max_depth': [20],
        'subsample': [1.0],
    }

    # Best parameters: {'max_depth': 20, 'n_estimators': 100, 'subsample': 1.0}, Best CV score: 0.7782

    print(f"Started tuning XGBoost classifier")
    try:
        model = xgb.XGBClassifier(random_state=42, n_jobs=MODEL_JOBS)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy', n_jobs=GRID_SEARCH_JOBS
        )
        grid_search.fit(X_train, y_train)
        print(f"Finished tuning XGBoost classifier")
        print(f"Best parameters: {grid_search.best_params_}, Best CV score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    except Exception as e:
        print(f"Error tuning XGBoost classifier: {e}")
        raise e



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
    # param_grid = {
    #     'n_estimators': [50, 100],
    #     'max_depth': [10, 20],
    #     'learning_rate': [0.01, 0.1],
    #     'subsample': [0.8, 1.0],
    # }

    param_grid = {
        'n_estimators': [100],
        'max_depth': [20],
        'learning_rate': [0.1],
        'subsample': [0.8],
    }

    print(f"Started tuning LightGBM regressor")
    try:
        model = lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=MODEL_JOBS)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='r2', n_jobs=GRID_SEARCH_JOBS
        )
        grid_search.fit(X_train, y_train)
        print(f"Finished tuning LightGBM regressor")
        print(f"Best parameters: {grid_search.best_params_}, Best CV score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    except Exception as e:
        print(f"Error tuning LightGBM regressor: {e}")
        raise e



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

    # Best parameters: {'learning_rate': 0.01, 'max_depth': 20, 'n_estimators': 100, 'subsample': 0.8}, Best CV score: 0.7886
    # param_grid = {
    #     'n_estimators': [50, 100],
    #     'max_depth': [10, 20],
    #     'learning_rate': [0.01, 0.1],
    #     'subsample': [0.8, 1.0],
    # }

    param_grid = {
        'n_estimators': [100],
        'max_depth': [20],
        'learning_rate': [0.01],
        'subsample': [0.8],
    }

    print(f"Started tuning LightGBM classifier")
    try:
        model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=MODEL_JOBS)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy', n_jobs=GRID_SEARCH_JOBS
        )
        grid_search.fit(X_train, y_train)
        print(f"Finished tuning LightGBM classifier")
        print(f"Best parameters: {grid_search.best_params_}, Best CV score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    except Exception as e:
        print(f"Error tuning LightGBM classifier: {e}")
        raise e



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
    # Best parameters: {'depth': 6, 'iterations': 50, 'l2_leaf_reg': 1, 'learning_rate': 0.1}
    # param_grid = {
    #     'iterations': [50, 100],
    #     'depth': [6, 10],
    #     'learning_rate': [0.01, 0.1],
    #     'l2_leaf_reg': [1, 3],
    # }

    param_grid = {
        'iterations': [50],
        'depth': [6],
        'learning_rate': [0.1],
        'l2_leaf_reg': [1],
    }

    print(f"Started tuning CatBoost regressor")
    try:
        model = cb.CatBoostRegressor(random_state=42, verbose=False, thread_count=MODEL_JOBS)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='r2', n_jobs=GRID_SEARCH_JOBS
        )
        grid_search.fit(X_train, y_train)
        print(f"Finished tuning CatBoost regressor")
        print(f"Best parameters: {grid_search.best_params_}, Best CV score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    except Exception as e:
        print(f"Error tuning CatBoost regressor: {e}")
        raise e



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
    # param_grid = {
    #     'iterations': [50, 100],
    #     'depth': [6, 10],
    #     'learning_rate': [0.01, 0.1],
    #     'l2_leaf_reg': [1, 3],
    # }

    param_grid = {
        'iterations': [50],
        'depth': [6],
        'learning_rate': [0.1],
        'l2_leaf_reg': [1],
    }

    print(f"Started tuning CatBoost classifier")
    try:
        model = cb.CatBoostClassifier(random_state=42, verbose=False, thread_count=MODEL_JOBS)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy', n_jobs=GRID_SEARCH_JOBS
        )
        grid_search.fit(X_train, y_train)
        print(f"Finished tuning CatBoost classifier")
        print(f"Best parameters: {grid_search.best_params_}, Best CV score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    except Exception as e:
        print(f"Error tuning CatBoost classifier: {e}")
        raise e 
