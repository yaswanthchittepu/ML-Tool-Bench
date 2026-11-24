from typing import Annotated, Sequence, NotRequired, Any, Union, Dict
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import pickle

# LangSmith configuration
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "mle"

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    scratchpad: NotRequired[Dict[str, Any]]
    
@tool
def read_data(filepath: str, save_name: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    Read CSV data into a pandas DataFrame.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file to load.

    save_name: str
       The key of the object in the scratchpad that stores variables, to save the object to 

    """
    df = pd.read_csv(filepath)
    # return Command(update={
    #     "dataframe": df,
    #     "messages": [
    #         ToolMessage(f"Updated dataframe", tool_call_id=tool_call_id)
    #     ]
    # })
    return Command(update={
        "scratchpad": {
            save_name: df  # <- this is how you store it under a key like "df1"
        },
        "messages": [
            ToolMessage(f"Stored DataFrame in scratchpad as '{name}'", tool_call_id=tool_call_id)
        ]
    })

@tool
def explore_dataset(load_name: str, state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    Explore the dataset present in the state and analyze all columns for feature engineering decisions.

    Parameters
    ----------

    load_name: str
       The name of the object in the scratchpad that stores variables, to load the object from


    Returns
    -------
    dict
        Dictionary containing dataset summary information.
        
    """
    df = state['scratchpad'][load_name]
    # Analyze columns
    column_info = {}
    for col in df.columns:
        col_data = df[col]
        dtype = str(col_data.dtype)
        
        # Basic statistics
        info = {
            'dtype': dtype,
            'missing_count': col_data.isnull().sum(),
            'missing_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
            'unique_count': col_data.nunique()
        }
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            info.update({
                'min': col_data.min(),
                'max': col_data.max(),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'type': 'numeric'
            })
        # Categorical/text columns
        elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
            info.update({
                'type': 'categorical',
                'most_common': col_data.mode().iloc[0] if not col_data.mode().empty else None,
                'avg_length': col_data.astype(str).str.len().mean()
            })
        # Datetime columns
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            info.update({
                'type': 'datetime',
                'min_date': col_data.min(),
                'max_date': col_data.max()
            })
        else:
            info['type'] = 'other'
        
        column_info[col] = info
    
    # Dataset summary
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'column_info': column_info,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    # return summary
    return Command(update={
        "messages": [
            ToolMessage(f"Dataset summary:\n{summary}", tool_call_id=tool_call_id)
        ]
    })


@tool
def clean_data(
    name: str, state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]
):
    """
    Clean the DataFrame present in the state by removing duplicates and filling missing values.

    Parameters
    ----------

    name: str
       The name of the object in the scratchpad that stores variables
    
    """

    df = state['scratchpad'][name].drop_duplicates()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
    return Command(update={
        "dataframe": df,
        "messages": [
            ToolMessage(f"Cleaned and updated dataframe", tool_call_id=tool_call_id)
        ]
    })

# @tool
# def feature_engineer(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]):
#     """
#     Perform feature engineering on the DataFrame present in the state.
    
#     Applies one-hot encoding to categorical columns and standard scaling
#     to numerical columns.
    
#     """
#     df = pd.get_dummies(state['dataframe'], drop_first=True)
#     scaler = StandardScaler()
#     num_cols = df.select_dtypes(include=[np.number]).columns
#     df[num_cols] = scaler.fit_transform(df[num_cols])
#     return Command(update={
#         "dataframe": df,
#         "messages": [
#             ToolMessage(f"Feature engineered and Updated dataframe", tool_call_id=tool_call_id)
#         ]
#     })

# @tool
# def split_data(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId], target_column: str, test_size: float=0.2, random_state: int=42):
#     """
#     Split the pandas dataframe in the state into training and test sets.

#     The training set will contain the features and the target (X_train, Y_train)

#     Likewise, the test set will contain the features and the target (X_train, Y_train)
    
#     Parameters (Not including the injected variables of state and tool_call_id)
#     ----------
#     target_column : str
#         The column to use as the target variable.
#     test_size : float, default=0.2
#         Proportion of data to use for testing.
#     random_state : int, default=42
#         Random seed for reproducibility.
        
#     """
#     X = state['dataframe'].drop(target_column, axis=1)
#     y = state['dataframe'][target_column]
#     X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
#     print(f"X_train shape: {X_train.shape}")
#     print(f"X_test shape: {X_test.shape}")
#     print(f"Y_train shape: {Y_train.shape}")
#     print(f"Y_test shape: {Y_test.shape}")
#     return Command(update={
#         "X_train": X_train,
#         "X_test": X_test, 
#         "Y_train": Y_train,
#         "Y_test": Y_test,
#         "messages": [ToolMessage(f"Split data into train/test sets", tool_call_id=tool_call_id)]
#     })


# ############## TRAINING MODELS ##############

# @tool
# def train_linear_regressor(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]):
#     """
#     Train a Linear Regression model on train data X_train, Y_train present in the state.

#     """
#     print(f"Started training linear regressor")
#     model = LinearRegression()
#     model.fit(state['X_train'], state['Y_train'])
#     print(f"Finished training linear regressor")
    
#     return Command(update={
#         "model": model,
#         "messages": [
#             ToolMessage(f"Trained and Updated linear regressor model", tool_call_id=tool_call_id)
#         ]
#     })

# @tool
# def train_logistic_regressor(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]):
#     """
#     Train a Logistic Regression model on train data X_train, Y_train present in the state.

#     """
#     print(f"Started training logistic regressor")
#     model = LogisticRegression(random_state=42, max_iter=1000)
#     model.fit(state['X_train'], state['Y_train'])
#     print(f"Finished training logistic regressor")
    
#     return Command(update={
#         "model": model,
#         "messages": [
#             ToolMessage(f"Trained and Updated logistic regressor model", tool_call_id=tool_call_id)
#         ]
#     })

# @tool
# def train_random_forest_regressor(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]):
#     """
#     Train a Random Forest Regressor model on train data X_train, Y_train present in the state.

#     """

#     # X = state['dataframe'].drop(target_column, axis=1)
#     # y = state['dataframe'][target_column]

#     print(f"Started training random forest regressor")
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(state['X_train'], state['Y_train'])
#     print(f"Finished training random forest regressor")

#     return Command(update={
#         "model": model,
#         "messages": [
#             ToolMessage(f"Trained and Updated model", tool_call_id=tool_call_id)
#         ]
#     })

# @tool
# def train_random_forest_classifier(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]):
#     """
#     Train a Random Forest Classifier model on train data X_train, Y_train present in the state.

#     """

#     print(f"Started training random forest classifier")
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(state['X_train'], state['Y_train'])
#     print(f"Finished training random forest classifier")
    
#     return Command(update={
#         "model": model,
#         "messages": [
#             ToolMessage(f"Trained and Updated dataframe", tool_call_id=tool_call_id)
#         ]
#     })

# @tool
# def train_xgboost_regressor(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]):
#     """
#     Train an XGBoost Regressor model on train data X_train, Y_train present in the state.

#     """
#     print(f"Started training XGBoost regressor")
#     model = xgb.XGBRegressor(n_estimators=100, random_state=42)
#     model.fit(state['X_train'], state['Y_train'])
#     print(f"Finished training XGBoost regressor")
    
#     return Command(update={
#         "model": model,
#         "messages": [
#             ToolMessage(f"Trained and Updated XGBoost regressor model", tool_call_id=tool_call_id)
#         ]
#     })

# @tool
# def train_xgboost_classifier(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]):
#     """
#     Train an XGBoost Classifier model on train data X_train, Y_train present in the state.

#     """
#     print(f"Started training XGBoost classifier")
#     model = xgb.XGBClassifier(n_estimators=100, random_state=42)
#     model.fit(state['X_train'], state['Y_train'])
#     print(f"Finished training XGBoost classifier")
    
#     return Command(update={
#         "model": model,
#         "messages": [
#             ToolMessage(f"Trained and Updated XGBoost classifier model", tool_call_id=tool_call_id)
#         ]
#     })

# ############## EVALUATING MODELS ##############

# @tool
# def evaluate_regression_model(state: Annotated[AgentState, InjectedState], eval_data_label: str = 'test'):
#     """
#     Evaluate a trained regression model on test data.

#     Parameters (Not including the state injected variable)
#     ----------
#     eval_data_label : str, default='test'
#         Label of the data to evaluate on.
#         Can be 'test' or 'train'
#         Will raise an error if the label is not 'test' or 'train'
        
#     Returns
#     -------
#     str
#         A string with the evaluation results.
        
#     """

#     if eval_data_label == 'test':
#         X_test = state['X_test']
#         y_test = state['Y_test']
#     elif eval_data_label == 'train':
#         X_test = state['X_train']
#         y_test = state['Y_train']
    
#     trained_model = state['model']
#     y_pred = trained_model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     # print(f"Test MSE: {mse:.4f}")
#     # print(f"Test R2: {r2:.4f}")
#     # return y_pred
#     return f"On the {eval_data_label} data, the model has a MSE of {mse:.4f} and an R2 score of {r2:.4f}"

# @tool
# def evaluate_classification_model(state: Annotated[AgentState, InjectedState], eval_data_label: str = 'test'):
#     """
#     Evaluate a trained classification model on test data.
    
#     Parameters (Not including the state injected variable)
#     ----------
#     eval_data_label : str, default='test'
#         Label of the data to evaluate on.
#         Can be 'test' or 'train'
#         Will raise an error if the label is not 'test' or 'train'
        
#     Returns
#     -------
#     str
#         A string with the evaluation results.
        
#     """
#     if eval_data_label == 'test':
#         X_test = state['X_test']
#         y_test = state['Y_test']
#     elif eval_data_label == 'train':
#         X_test = state['X_train']
#         y_test = state['Y_train']
    
#     trained_model = state['model']
#     y_pred = trained_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     cls_report = classification_report(y_test, y_pred)
#     # print(f"Test Accuracy: {accuracy:.4f}")
#     # print(f"Classification Report:")
#     # print(classification_report(y_test, y_pred))
#     # return y_pred
#     return f"On the {eval_data_label} data, the model has an accuracy of {accuracy:.4f}. The classification report is: {cls_report}"


# # ============================================================================
# # HYPERPARAMETER TUNING FUNCTIONS
# # ============================================================================


# @tool
# def tune_logistic_regressor(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId], cv: int = 5):
#     """
#     Perform hyperparameter tuning for Logistic Regression using GridSearchCV.
    
#     Parameters (Not including the state injected variable)
#     ----------
#     cv : int, default=5
#         Number of cross-validation folds.
        
#     Returns
#     -------
#     str
#         A string with the tuning results.
        
#     """
    
    
#     X_train = state['X_train']
#     y_train = state['Y_train']
    
#     param_grid = {
#         'C': [0.001, 0.01, 0.1, 1, 10, 100],
#         'penalty': ['l1', 'l2'],
#     }
    
#     model = LogisticRegression(random_state=42)
#     grid_search = GridSearchCV(
#         model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
#     )
#     grid_search.fit(X_train, y_train)
    
#     best_model = grid_search.best_estimator_
    
#     return Command(update={
#         "model": best_model,
#         "messages": [
#             ToolMessage(f"Tuned Logistic Regression. Best parameters: {grid_search.best_params_}, Best CV score: {grid_search.best_score_:.4f}", tool_call_id=tool_call_id)
#         ]
#     })

# @tool
# def tune_random_forest_regressor(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId], cv: int = 5):
#     """
#     Perform hyperparameter tuning for Random Forest Regressor using GridSearchCV.

#     Parameters (Not including the state injected variable)
#     ----------
#     cv : int, default=5
#         Number of cross-validation folds.

#     Returns
#     -------
#     str
#         A string with the tuning results.
#     """
#     X_train = state['X_train']
#     y_train = state['Y_train']

#     param_grid = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [10, 20, None],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4],
#         'max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0],
#         "bootstrap": [True, False],
#         "max_samples": [0.5, 0.75, 1.0],
#     }

#     model = RandomForestRegressor(random_state=42)
#     grid_search = GridSearchCV(
#         model, param_grid, cv=cv, scoring='r2', n_jobs=-1
#     )
#     grid_search.fit(X_train, y_train)

#     best_model = grid_search.best_estimator_

#     return Command(update={
#         "model": best_model,
#         "messages": [
#             ToolMessage(
#                 f"Tuned Random Forest Regressor. Best parameters: {grid_search.best_params_}, Best CV score: {grid_search.best_score_:.4f}",
#                 tool_call_id=tool_call_id
#             )
#         ]
#     })

# @tool
# def tune_random_forest_classifier(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId], cv: int = 5):
#     """
#     Perform hyperparameter tuning for Random Forest Classifier using GridSearchCV.

#     Parameters (Not including the state injected variable)
#     ----------
#     cv : int, default=5
#         Number of cross-validation folds.

#     Returns
#     -------
#     str
#         A string with the tuning results.
#     """
#     X_train = state['X_train']
#     y_train = state['Y_train']

#     param_grid = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [10, 20, None],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4],
#         'max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0],
#         "bootstrap": [True, False],
#         "max_samples": [0.5, 0.75, 1.0],
#     }

#     model = RandomForestClassifier(random_state=42)
#     grid_search = GridSearchCV(
#         model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
#     )
#     grid_search.fit(X_train, y_train)

#     best_model = grid_search.best_estimator_

#     return Command(update={
#         "model": best_model,
#         "messages": [
#             ToolMessage(
#                 f"Tuned Random Forest Classifier. Best parameters: {grid_search.best_params_}, Best CV score: {grid_search.best_score_:.4f}",
#                 tool_call_id=tool_call_id
#             )
#         ]
#     })

# @tool
# def tune_xgboost_regressor(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId], cv: int =5):
#     """
#     Perform hyperparameter tuning for XGBoost Regressor using GridSearchCV.

#     Parameters (Not including the state injected variable)
#     ----------
#     cv : int, default=5
#         Number of cross-validation folds.

#     Returns
#     -------
#     str
#         A string with the tuning results.
#     """
#     X_train = state['X_train']
#     y_train = state['Y_train']

#     param_grid = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [3, 6, 9, 12, 15],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'subsample': [0.8, 0.9, 1.0],
#         'reg_lambda': [0.0, 0.01, 0.1, 1.0],
#     }

#     model = xgb.XGBRegressor(random_state=42)
#     grid_search = GridSearchCV(
#         model, param_grid, cv=cv, scoring='r2', n_jobs=-1
#     )
#     grid_search.fit(X_train, y_train)

#     best_model = grid_search.best_estimator_

#     return Command(update={
#         "model": best_model,
#         "messages": [
#             ToolMessage(
#                 f"Tuned XGBoost Regressor. Best parameters: {grid_search.best_params_}, Best CV score: {grid_search.best_score_:.4f}",
#                 tool_call_id=tool_call_id
#             )
#         ]
#     })

# @tool
# def tune_xgboost_classifier(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId], cv: int =5):
#     """
#     Perform hyperparameter tuning for XGBoost Classifier using GridSearchCV.

#     Parameters (Not including the state injected variable)
#     ----------
#     cv : int, default=5
#         Number of cross-validation folds.

#     Returns
#     -------
#     str
#         A string with the tuning results.
#     """
#     X_train = state['X_train']
#     y_train = state['Y_train']

#     param_grid = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [3, 6, 9, 12, 15],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'subsample': [0.8, 0.9, 1.0],
#         'reg_lambda': [0.0, 0.01, 0.1, 1.0],
#     }

#     model = xgb.XGBClassifier(random_state=42)
#     grid_search = GridSearchCV(
#         model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
#     )
#     grid_search.fit(X_train, y_train)

#     best_model = grid_search.best_estimator_

#     return Command(update={
#         "model": best_model,
#         "messages": [
#             ToolMessage(
#                 f"Tuned XGBoost Classifier. Best parameters: {grid_search.best_params_}, Best CV score: {grid_search.best_score_:.4f}",
#                 tool_call_id=tool_call_id
#             )
#         ]
#     })


# @tool
# def save_model(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId], filepath: str = 'model.pkl'):
#     """
#     Save the trained model from state to disk using pickle.
    
#     Parameters (Not including the state injected variable)
#     ----------
#     filepath : str
#         Path where to save the model file (e.g., 'model.pkl')
        
#     Returns
#     -------
#     str
#         Confirmation message with the save location.
        
#     """
    
#     if 'model' not in state:
#         return "No model found in state. Please train a model first."
    
#     with open(filepath, 'wb') as f:
#         pickle.dump(state['model'], f)
    
#     return Command(update={
#         "messages": [
#             ToolMessage(f"Model saved successfully to {filepath}", tool_call_id=tool_call_id)
#         ]
#     })


# @tool
# def load_model(filepath: str, tool_call_id: Annotated[str, InjectedToolCallId]):
#     """
#     Load a trained model from disk using pickle.
    
#     Parameters
#     ----------
#     filepath : str
#         Path to the saved model file (e.g., 'model.pkl')
        
#     Returns
#     -------
#     str
#         Confirmation message with the load location.
        
#     """
    
#     try:
#         with open(filepath, 'rb') as f:
#             model = pickle.load(f)
        
#         return Command(update={
#             "model": model,
#             "messages": [
#                 ToolMessage(f"Model loaded successfully from {filepath}", tool_call_id=tool_call_id)
#             ]
#         })
#     except FileNotFoundError:
#         return f"Model file not found at {filepath}"
#     except Exception as e:
#         return f"Error loading model: {str(e)}"


# tools = [read_data, explore_dataset,clean_data, feature_engineer, split_data, train_linear_regressor, train_logistic_regressor, train_random_forest_regressor, train_random_forest_classifier, train_xgboost_regressor, train_xgboost_classifier, tune_logistic_regressor, tune_random_forest_regressor, tune_random_forest_classifier, tune_xgboost_regressor, tune_xgboost_classifier, evaluate_regression_model, evaluate_classification_model, save_model, load_model]

tools = [read_data, explore_dataset]

model = ChatOpenAI(model="gpt-4o-mini",
                    base_url='https://pluto-prod-hawang-llm-proxy-9qtfav-0-4000.colligo.dev',
                    api_key=os.environ['OPENAI_API_KEY']).bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are a Data Scientist. You are given a task to build a machine learning model. You will be given a dataset and you will need to build a model to predict the target variable. You will be given a set of tools to help you with the task. \
        IMPORTANT: Call only ONE tool at a time. Do not call multiple training or tuning tools simultaneously. Work sequentially - first load data, then clean, then engineer features, then split data, then train ONE model, then evaluate that model. If you want to try different models, train them one at a time."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState): 
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6.")]}
# print_stream(app.stream(inputs, stream_mode="values"))

human_query = """
Read the data from the train.csv file in the data folder.
"""

# human_query = """

# Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery. We've received a transmission from four lightyears away and things aren't looking good.

# The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.

# While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!
# Read the data from the train.csv file in the data folder.

# To help rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.

# Help save them and change history!

# In this competition your task is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. To help you make these predictions, you're given a set of personal records recovered from the ship's damaged computer system.

# File and Data Field Descriptions
# train.csv - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
# PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
# CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# Destination - The planet the passenger will be debarking to.
# Age - The age of the passenger.
# VIP - Whether the passenger has paid for special VIP service during the voyage.
# RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# Name - The first and last names of the passenger.
# Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

# The data is in the data folder and is named train.csv. 

# You have to provide me the best possible model you can for the task, using the tools provided to you. 

# You might need to experiment with different model and hyperparameter tuning to find the best model.

# Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves folder.

# """
# Give me the evaluation resuls, that show the goodness of the model fit to the data. Also save the model to the model_saves folder.

inputs = {"messages": [("user", f"{human_query}")]}
response = app.invoke(inputs, config={"recursion_limit": 50})

print(response['messages'][-1].content)
# current_state = {"messages": []}

# while True:
#     user_input = input("\nYou: ")
#     if user_input.lower() in ['quit', 'exit', 'q']:
#         print("Goodbye!")
#         break

#     # Add the new user message to the ongoing state
#     current_state["messages"].append(HumanMessage(content=user_input))
    
#     # Run the app
#     response = app.invoke(current_state)

#     # Update the full state
#     current_state = response

#     # Show assistant message
#     print(f"\n\n\n {response} \n\n\n")
#     print(f"Assistant: {response['messages'][-1].content}")