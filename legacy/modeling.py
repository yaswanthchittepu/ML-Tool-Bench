import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingRegressor, StackingClassifier, VotingRegressor, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

############## TRAINING MODELS ##############

# def train_linear_regressor(X_train: pd.DataFrame, Y_train: pd.Series, params: dict) -> LinearRegression:
#     """
#     Train a Linear Regression model on train data.

#     Parameters
#     ----------
#     X_train : pd.DataFrame
#         Training features
#     Y_train : pd.Series
#         Training target
#     params: dict
#         Additional parameters to pass to sklearn.linear_model.LinearRegression

#     Returns
#     -------
#     LinearRegression
#         Trained linear regression model
#     """
#     print(f"Started training linear regressor")
#     model = LinearRegression(**params)
#     model.fit(X_train, Y_train)
#     print(f"Finished training linear regressor")
#     return model


# def train_logistic_regressor(X_train: pd.DataFrame, Y_train: pd.Series, params: dict) -> LogisticRegression:
#     """
#     Train a Logistic Regression model on train data.

#     Parameters
#     ----------
#     X_train : pd.DataFrame
#         Training features
#     Y_train : pd.Series
#         Training target
#     params: dict
#         Additional parameters to pass to sklearn.linear_model.LogisticRegression

#     Returns
#     -------
#     LogisticRegression
#         Trained logistic regression model
#     """
#     print(f"Started training logistic regressor")
#     model = LogisticRegression(**params)
#     model.fit(X_train, Y_train)
#     print(f"Finished training logistic regressor")
#     return model


# def train_random_forest_regressor(X_train: pd.DataFrame, Y_train: pd.Series, params: dict) -> RandomForestRegressor:
#     """
#     Train a Random Forest Regressor model on train data.

#     Parameters
#     ----------
#     X_train : pd.DataFrame
#         Training features
#     Y_train : pd.Series
#         Training target
#     params: dict
#         Additional parameters to pass to sklearn.ensemble.RandomForestRegressor

#     Returns
#     -------
#     RandomForestRegressor
#         Trained random forest regressor model
#     """
#     print(f"Started training random forest regressor")
#     model = RandomForestRegressor(**params)
#     model.fit(X_train, Y_train)
#     print(f"Finished training random forest regressor")
#     return model


# def train_random_forest_classifier(X_train: pd.DataFrame, Y_train: pd.Series, params: dict) -> RandomForestClassifier:
#     """
#     Train a Random Forest Classifier model on train data.

#     Parameters
#     ----------
#     X_train : pd.DataFrame
#         Training features
#     Y_train : pd.Series
#         Training target
#     params: dict
#         Additional parameters to pass to sklearn.ensemble.RandomForestClassifier

#     Returns
#     -------
#     RandomForestClassifier
#         Trained random forest classifier model
#     """
#     print(f"Started training random forest classifier")
#     model = RandomForestClassifier(**params)
#     model.fit(X_train, Y_train)
#     print(f"Finished training random forest classifier")
#     return model


# def train_xgboost_regressor(X_train: pd.DataFrame, Y_train: pd.Series, params: dict) -> xgb.XGBRegressor:
#     """
#     Train an XGBoost Regressor model on train data.

#     Parameters
#     ----------
#     X_train : pd.DataFrame
#         Training features
#     Y_train : pd.Series
#         Training target
#     params: dict
#         Additional parameters to pass to xgboost.XGBRegressor

#     Returns
#     -------
#     xgb.XGBRegressor
#         Trained XGBoost regressor model
#     """
#     print(f"Started training XGBoost regressor")
#     model = xgb.XGBRegressor(**params)
#     model.fit(X_train, Y_train)
#     print(f"Finished training XGBoost regressor")
#     return model


# def train_xgboost_classifier(X_train: pd.DataFrame, Y_train: pd.Series, params: dict) -> xgb.XGBClassifier:
#     """
#     Train an XGBoost Classifier model on train data.

#     Parameters
#     ----------
#     X_train : pd.DataFrame
#         Training features
#     Y_train : pd.Series
#         Training target
#     params: dict
#         Additional parameters to pass to xgboost.XGBClassifier

#     Returns
#     -------
#     xgb.XGBClassifier
#         Trained XGBoost classifier model
#     """
#     print(f"Started training XGBoost classifier")
#     model = xgb.XGBClassifier(**params)
#     model.fit(X_train, Y_train)
#     print(f"Finished training XGBoost classifier")
#     return model


# def train_lightgbm_regressor(X_train: pd.DataFrame, Y_train: pd.Series, params: dict) -> lgb.LGBMRegressor:
#     """
#     Train a LightGBM Regressor model on train data.

#     Parameters
#     ----------
#     X_train : pd.DataFrame
#         Training features
#     Y_train : pd.Series
#         Training target
#     params: dict
#         Additional parameters to pass to lightgbm.LGBMRegressor

#     Returns
#     -------
#     lgb.LGBMRegressor
#         Trained LightGBM regressor model
#     """
#     print(f"Started training LightGBM regressor")
#     model = lgb.LGBMRegressor(**params)
#     model.fit(X_train, Y_train)
#     print(f"Finished training LightGBM regressor")
#     return model


# def train_lightgbm_classifier(X_train: pd.DataFrame, Y_train: pd.Series, params: dict) -> lgb.LGBMClassifier:
#     """
#     Train a LightGBM Classifier model on train data.

#     Parameters
#     ----------
#     X_train : pd.DataFrame
#         Training features
#     Y_train : pd.Series
#         Training target
#     params: dict
#         Additional parameters to pass to lightgbm.LGBMClassifier

#     Returns
#     -------
#     lgb.LGBMClassifier
#         Trained LightGBM classifier model
#     """
#     print(f"Started training LightGBM classifier")
#     model = lgb.LGBMClassifier(**params)
#     model.fit(X_train, Y_train)
#     print(f"Finished training LightGBM classifier")
#     return model


# def train_catboost_regressor(X_train: pd.DataFrame, Y_train: pd.Series, params: dict) -> cb.CatBoostRegressor:
#     """
#     Train a CatBoost Regressor model on train data.

#     Parameters
#     ----------
#     X_train : pd.DataFrame
#         Training features
#     Y_train : pd.Series
#         Training target
#     params: dict
#         Additional parameters to pass to catboost.CatBoostRegressor

#     Returns
#     -------
#     cb.CatBoostRegressor
#         Trained CatBoost regressor model
#     """
#     print(f"Started training CatBoost regressor")
#     model = cb.CatBoostRegressor(**params)
#     model.fit(X_train, Y_train)
#     print(f"Finished training CatBoost regressor")
#     return model


# def train_catboost_classifier(X_train: pd.DataFrame, Y_train: pd.Series, params: dict) -> cb.CatBoostClassifier:
#     """
#     Train a CatBoost Classifier model on train data.

#     Parameters
#     ----------
#     X_train : pd.DataFrame
#         Training features
#     Y_train : pd.Series
#         Training target
#     params: dict
#         Additional parameters to pass to catboost.CatBoostClassifier

#     Returns
#     -------
#     cb.CatBoostClassifier
#         Trained CatBoost classifier model
#     """
#     print(f"Started training CatBoost classifier")
#     model = cb.CatBoostClassifier(**params)
#     model.fit(X_train, Y_train)
#     print(f"Finished training CatBoost classifier")
#     return model


# def create_stacking_regressor(base_models: list, meta_model=None, cv: int = 5) -> StackingRegressor:
#     """
#     Create a stacking regressor with multiple base models.
    
#     Parameters
#     ----------
#     base_models : list
#         List of (name, model) tuples for base models
#     meta_model : estimator, optional
#         Meta-model for final prediction. If None, uses LinearRegression
#     cv : int, default=5
#         Number of cross-validation folds for meta-features
        
#     Returns
#     -------
#     StackingRegressor
#         Trained stacking regressor model
#     """
#     if meta_model is None:
#         meta_model = LinearRegression()
    
#     print(f"Creating stacking regressor with {len(base_models)} base models")
#     for name, model in base_models:
#         print(f"  - {name}: {type(model).__name__}")
    
#     stacking_model = StackingRegressor(
#         estimators=base_models,
#         final_estimator=meta_model,
#         cv=cv,
#         n_jobs=-1
#     )
    
#     return stacking_model


# def create_stacking_classifier(base_models: list, meta_model=None, cv: int = 5) -> StackingClassifier:
#     """
#     Create a stacking classifier with multiple base models.
    
#     Parameters
#     ----------
#     base_models : list
#         List of (name, model) tuples for base models
#     meta_model : estimator, optional
#         Meta-model for final prediction. If None, uses LogisticRegression
#     cv : int, default=5
#         Number of cross-validation folds for meta-features
        
#     Returns
#     -------
#     StackingClassifier
#         Trained stacking classifier model
#     """
#     if meta_model is None:
#         meta_model = LogisticRegression(random_state=42)
    
#     print(f"Creating stacking classifier with {len(base_models)} base models")
#     for name, model in base_models:
#         print(f"  - {name}: {type(model).__name__}")
    
#     stacking_model = StackingClassifier(
#         estimators=base_models,
#         final_estimator=meta_model,
#         cv=cv,
#         n_jobs=-1
#     )
    
#     return stacking_model


# def train_stacking_regressor(X_train: pd.DataFrame, Y_train: pd.Series, base_models: list, 
#                            meta_model=None, cv: int = 5) -> StackingRegressor:
#     """
#     Train a stacking regressor with multiple base models.
    
#     Parameters
#     ----------
#     X_train : pd.DataFrame
#         Training features
#     Y_train : pd.Series
#         Training target
#     base_models : list
#         List of (name, model) tuples for base models
#     meta_model : estimator, optional
#         Meta-model for final prediction. If None, uses LinearRegression
#     cv : int, default=5
#         Number of cross-validation folds for meta-features
        
#     Returns
#     -------
#     StackingRegressor
#         Trained stacking regressor model
#     """
#     print(f"Started training stacking regressor")
#     stacking_model = create_stacking_regressor(base_models, meta_model, cv)
#     stacking_model.fit(X_train, Y_train)
#     print(f"Finished training stacking regressor")
#     return stacking_model


# def train_stacking_classifier(X_train: pd.DataFrame, Y_train: pd.Series, base_models: list, 
#                             meta_model=None, cv: int = 5) -> StackingClassifier:
#     """
#     Train a stacking classifier with multiple base models.
    
#     Parameters
#     ----------
#     X_train : pd.DataFrame
#         Training features
#     Y_train : pd.Series
#         Training target
#     base_models : list
#         List of (name, model) tuples for base models
#     meta_model : estimator, optional
#         Meta-model for final prediction. If None, uses LogisticRegression
#     cv : int, default=5
#         Number of cross-validation folds for meta-features
        
#     Returns
#     -------
#     StackingClassifier
#         Trained stacking classifier model
#     """
#     print(f"Started training stacking classifier")
#     stacking_model = create_stacking_classifier(base_models, meta_model, cv)
#     stacking_model.fit(X_train, Y_train)
#     print(f"Finished training stacking classifier")
#     return stacking_model


# def create_voting_regressor(base_models: list, voting: str = 'hard') -> VotingRegressor:
#     """
#     Create a voting regressor with multiple base models.
    
#     Parameters
#     ----------
#     base_models : list
#         List of (name, model) tuples for base models
#     voting : str, default='hard'
#         Voting strategy ('hard' or 'soft')
        
#     Returns
#     -------
#     VotingRegressor
#         Voting regressor model
#     """
#     print(f"Creating voting regressor with {len(base_models)} base models")
#     for name, model in base_models:
#         print(f"  - {name}: {type(model).__name__}")
    
#     voting_model = VotingRegressor(
#         estimators=base_models,
#         n_jobs=-1
#     )
    
#     return voting_model


# def create_voting_classifier(base_models: list, voting: str = 'hard') -> VotingClassifier:
#     """
#     Create a voting classifier with multiple base models.
    
#     Parameters
#     ----------
#     base_models : list
#         List of (name, model) tuples for base models
#     voting : str, default='hard'
#         Voting strategy ('hard' or 'soft')
        
#     Returns
#     -------
#     VotingClassifier
#         Voting classifier model
#     """
#     print(f"Creating voting classifier with {len(base_models)} base models")
#     for name, model in base_models:
#         print(f"  - {name}: {type(model).__name__}")
    
#     voting_model = VotingClassifier(
#         estimators=base_models,
#         voting=voting,
#         n_jobs=-1
#     )
    
#     return voting_model


# def train_voting_regressor(X_train: pd.DataFrame, Y_train: pd.Series, base_models: list, 
#                           voting: str = 'hard') -> VotingRegressor:
#     """
#     Train a voting regressor with multiple base models.
    
#     Parameters
#     ----------
#     X_train : pd.DataFrame
#         Training features
#     Y_train : pd.Series
#         Training target
#     base_models : list
#         List of (name, model) tuples for base models
#     voting : str, default='hard'
#         Voting strategy ('hard' or 'soft')
        
#     Returns
#     -------
#     VotingRegressor
#         Trained voting regressor model
#     """
#     print(f"Started training voting regressor")
#     voting_model = create_voting_regressor(base_models, voting)
#     voting_model.fit(X_train, Y_train)
#     print(f"Finished training voting regressor")
#     return voting_model


# def train_voting_classifier(X_train: pd.DataFrame, Y_train: pd.Series, base_models: list, 
#                            voting: str = 'hard') -> VotingClassifier:
#     """
#     Train a voting classifier with multiple base models.
    
#     Parameters
#     ----------
#     X_train : pd.DataFrame
#         Training features
#     Y_train : pd.Series
#         Training target
#     base_models : list
#         List of (name, model) tuples for base models
#     voting : str, default='hard'
#         Voting strategy ('hard' or 'soft')
        
#     Returns
#     -------
#     VotingClassifier
#         Trained voting classifier model
#     """
#     print(f"Started training voting classifier")
#     voting_model = create_voting_classifier(base_models, voting)
#     voting_model.fit(X_train, Y_train)
#     print(f"Finished training voting classifier")
#     return voting_model


# def ensemble_predictions(models: list, X_data: pd.DataFrame, method: str = 'mean') -> pd.Series:
#     """
#     Ensemble predictions from multiple models using different methods.
    
#     Parameters
#     ----------
#     models : list
#         List of trained models
#     X_data : pd.DataFrame
#         Data to make predictions on
#     method : str, default='mean'
#         Ensemble method ('mean', 'median', 'max', 'min')
        
#     Returns
#     -------
#     pd.Series
#         Ensembled predictions
#     """
#     print(f"Making ensemble predictions using {len(models)} models with {method} method")
    
#     predictions = []
#     for i, model in enumerate(models):
#         pred = model.predict(X_data)
#         predictions.append(pred)
#         print(f"  Model {i+1}: {type(model).__name__}")
    
#     predictions_df = pd.DataFrame(predictions).T
    
#     if method == 'mean':
#         ensemble_pred = predictions_df.mean(axis=1)
#     elif method == 'median':
#         ensemble_pred = predictions_df.median(axis=1)
#     elif method == 'max':
#         ensemble_pred = predictions_df.max(axis=1)
#     elif method == 'min':
#         ensemble_pred = predictions_df.min(axis=1)
#     else:
#         raise ValueError(f"Unknown ensemble method: {method}")
    
#     print(f"Ensemble predictions completed using {method} method")
#     return ensemble_pred


# def ensemble_classification_predictions(models: list, X_data: pd.DataFrame, method: str = 'majority') -> pd.Series:
#     """
#     Ensemble classification predictions from multiple models.
    
#     Parameters
#     ----------
#     models : list
#         List of trained classification models
#     X_data : pd.DataFrame
#         Data to make predictions on
#     method : str, default='majority'
#         Ensemble method ('majority', 'weighted')
        
#     Returns
#     -------
#     pd.Series
#         Ensembled predictions
#     """
#     print(f"Making ensemble classification predictions using {len(models)} models with {method} method")
    
#     predictions = []
#     for i, model in enumerate(models):
#         pred = model.predict(X_data)
#         predictions.append(pred)
#         print(f"  Model {i+1}: {type(model).__name__}")
    
#     predictions_df = pd.DataFrame(predictions).T
    
#     if method == 'majority':
#         # Take the most common prediction for each sample
#         ensemble_pred = predictions_df.mode(axis=1).iloc[:, 0]
#     elif method == 'weighted':
#         # For now, simple mean (can be extended with weights)
#         ensemble_pred = predictions_df.mean(axis=1).round().astype(int)
#     else:
#         raise ValueError(f"Unknown ensemble method: {method}")
    
#     print(f"Ensemble classification predictions completed using {method} method")
#     return ensemble_pred
