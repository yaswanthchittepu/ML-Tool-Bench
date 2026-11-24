from collections import defaultdict
import feature_engineering as fe
import hyperparameter_tuning as hpt
import evaluation as ev
import mle_utils as ut
import modeling as md
import loading as ld
from func_to_tool_decorator_hierarchical import make_get_and_set_tool, make_get_tool, make_set_tool, override_tool
from challenges import CHALLENGES

from typing import Annotated, Sequence, NotRequired, Any, Union, Dict
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage # The foundational class for all message types in LangGraph
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.callbacks import get_openai_callback

import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv
import sys, ipdb, os, json
import inspect
import warnings
import uuid

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    scratchpad: NotRequired[Dict[str, Any]]
    current_stage: NotRequired[str]
    plan: Dict[str, Any]
    

if __name__ == "__main__":

    load_dotenv()
    
    allowed_functions = set(['read_data', 'create_conditional_feature', 'concatenate_train_test', 'fillna_with_median', 'fillna_with_mode',
        'extract_string_pattern', 'split_string_column', 'create_group_aggregation', 'drop_feature', 'cast_columns',
        'normalize_all_numerical_columns', 'encode_all_categorical_columns', 'split_combined_into_train_test',
        'drop_feature', 'get_features', 'tune_catboost_classifier', 'tune_random_forest_classifier',
        'tune_lightgbm_classifier', 'tune_logistic_classifier', 'tune_xgboost_classifier', 'predict_target', 
        'rename_feature', 'concatenate_dataframes', 'save_dataframe_to_csv'
    ])

    # Load the decorator mapping from JSON file
    with open("function_to_decorator_map.json", "r") as f:
        decorator_map = json.load(f)
    
    # Load the stage mapping from JSON file
    with open("stage_function_mapping_compact.json", "r") as f:
        stage_mapping = json.load(f)
    
    # Apply decorators to functions
    print("\n" + "="*80)
    print("APPLYING DECORATORS TO FUNCTIONS")
    print("="*80)
    
    # Module mapping
    module_map = {
        'feature_engineering': fe,
        'loading': ld,
        'modeling': md,
        'hyperparameter_tuning': hpt,
        'evaluation': ev,
        'mle_utils': ut
    }
    
    # Decorator function mapping
    decorator_func_map = {
        'make_get_and_set_tool': make_get_and_set_tool,
        'make_get_tool': make_get_tool,
        'make_set_tool': make_set_tool,
        'override_tool': override_tool
    }
    
    # Apply decorators
    decorated_functions = {}
    for module_name, functions in decorator_map.items():
        module = module_map[module_name]
        decorated_functions[module_name] = {}
        
        for func_name, decorator_name in functions.items():
            # if(func_name not in allowed_functions):
            #     continue
            try:
                # Get the original function
                original_func = getattr(module, func_name)
                
                # Get the decorator function
                decorator_func = decorator_func_map[decorator_name]
                
                # Apply the decorator
                decorated_func = decorator_func(original_func)
                
                # Store the decorated function
                decorated_functions[module_name][func_name] = decorated_func
                
                print(f"✓ Applied {decorator_name} to {module_name}.{func_name}")
                
            except Exception as e:
                print(f"✗ Failed to apply {decorator_name} to {module_name}.{func_name}: {e}")
    
    print(f"\nSuccessfully decorated {sum(len(funcs) for funcs in decorated_functions.values())} functions")

    # Merge all decorated functions into a single dictionary
    all_decorated_functions = {}
    for module_dict in decorated_functions.values():
        all_decorated_functions.update(module_dict)

    # Apply @tool decorator to all decorated functions
    print("\n" + "="*80)
    print("APPLYING @TOOL DECORATOR TO ALL FUNCTIONS")
    print("="*80)
    
    tool_functions = {}
    for func_name, decorated_func in all_decorated_functions.items():
        try:
            # Apply @tool decorator
            tool_func = tool(decorated_func)
            tool_functions[func_name] = tool_func
            print(f"✓ Applied @tool to {func_name}")
        except Exception as e:
            print(f"✗ Failed to apply @tool to {func_name}: {e}")
    
    print(f"\nSuccessfully created {len(tool_functions)} tool functions")

    # Create stage-specific tool sets
    stage_tools = {}
    for stage_name, stage_info in stage_mapping.items():
        stage_tools[stage_name] = []
        for func_name in stage_info["functions"]:
            if func_name in tool_functions:
                stage_tools[stage_name].append(tool_functions[func_name])
        print(f"Stage '{stage_name}': {len(stage_tools[stage_name])} tools")

    # ------------------------------------------------------------
    # Model Definition - Single Model with All Tools
    # ------------------------------------------------------------

    # Create single model with all tools
    all_tools = list(tool_functions.values())
    bind_kwargs = {'parallel_tool_calls': False} if('o3' not in os.environ['OPENAI_MODEL']) else {}

    models = {
            stage_name: ChatOpenAI(model=os.environ['OPENAI_MODEL'],
                            base_url=os.environ['OPENAI_BASE_URL'],
                            api_key=os.environ['OPENAI_API_KEY'],
                            temperature=float(os.environ['TEMPERATURE'])
                            ).bind_tools(
                                stage_tools[stage_name], 
                                **bind_kwargs
                            ) 
                        for stage_name in stage_mapping.keys()
    }
    # ------------------------------
    # Graph Definition with Stage Routing
    # ------------------------------
    mid = uuid.uuid4()
    save_directory = f"submissions/{os.environ['COMPETITION_NAME']}/hierarchical_react"
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(f"model_saves/{os.environ['COMPETITION_NAME']}/", exist_ok=True)
    competition_description = HumanMessage(content=CHALLENGES[os.environ['COMPETITION_NAME']].format(save_directory=save_directory, save_file_name=f"hierarchical_react_agent_{mid}.csv"))

    # Most Importantly, save your predictions for the test data, as a csv, to the submissions folder. The csv should have the following columns: PassengerId - Id for each passenger in the test set.
    # Transported - The predicted value. For each passenger, predict either True or False. Name the file as hierarchical_react_agent_{model_name}.csv. Look at the sample_submission.csv file in the data folder for an example of the format the results should be in. \n\n

    
    # stage_descriptions = {
    #     "data_loading_eda": f"You are a Data Scientist in the Data Loading and Exploratory Data Analysis stage of solving a Kaggle challenge. Load the train and test data and explore the data thoroughly. Good understanding of the data is crucial for the next stages. This stage ends when you have successfully loaded the train and test data and have explored the data thoroughly. You are allowed a maximum of {max_avail_tool_calls['data_loading_eda']} tool calls in this stage",
    #     "data_cleaning": f"You are a Data Scientist in the Data Cleaning stage of solving a Kaggle challenge. Clean and preprocess the data. This stage involves handling missing values, outliers, casting features to correct formats and other data quality issues. You are allowed to be innovative and use your understanding of the data to fill out missing values. This stage ends when the missing values and outliers are handled and features have been cast to the formats needed for downstream modeling. After the end of this stage, no missing values should be present in the data. You are recommended to apply these methods on a combined train and test dataset, to ensure consistent application across both partitions. You are not allowed to drop any rows from the test data, as the predictions will be needed for all datapoints. You are allowed a maximum of {max_avail_tool_calls['data_cleaning']} tool calls in this stage",
    #     "feature_engineering": f"You are a Data Scientist in the Feature Engineering stage of solving a Kaggle challenge. Create new features, or delete unimportant features or transform existing features as needed. This stage involves creating new features, dropping features, and other feature engineering techniques. Use your thorough understanding of data from earlier stages to aid your decisions. You are allowed to be innovative and use your understanding of the data to create new features. This stage ends when the models feels that the features are good enough for modeling, and categorical and numerical features have been properly encoded. After the end of this stage, all the features should be either int or float.  You are recommended to apply these methods on a combined train and test dataset, to ensure consistent application across both partitions.  You are not allowed to drop any rows from the test data, as the predictions will be needed for all datapoints. You are allowed a maximum of {max_avail_tool_calls['feature_engineering']} tool calls in this stage",
    #     "modeling": f"You are a Data Scientist in the Modeling stage of solving a Kaggle challenge. Train and tune models. You might need to experiment with different model choices and properly tune your hyperparameters to get good performance. Use the provided evaluation tools to evaluate your trained models to aid your decisions. This stage ends when the agent has successfully explored all available modeling and hyperparameter choices, and has found the best model it can from the given tools. You are allowed a maximum of {max_avail_tool_calls['modeling']} tool calls in this stage",
    #     # "evaluation": f"You are a Data Scientist in the Evaluation stage of solving a Kaggle challenge. Evaluate model performance.",
    #     "create_submission": f"You are a Data Scientist in the Create Submission stage. Save results and create submission files. The submission files should absolutely adhere to the format specified by the user. This stage ends when the submission files are created and saved to disk as per user instructions. You are allowed a maximum of {max_avail_tool_calls['create_submission']} tool calls in this stage"
    # }

    stage_descriptions = {
        "train_data_loading": f"You are a Data Scientist in the Data Loading stage of solving a Kaggle challenge, using only the tools available to you. Load the train and test data and explore the data thoroughly. This stage ends when you have loaded the train data successfully.",
        "test_data_loading": f"You are a Data Scientist in the Data Loading stage of solving a Kaggle challenge, using only the tools available to you. This stage ends when you have loaded the test data successfully.",
        "combine_train_test": f"You are a Data Scientist in the Data Loading stage of solving a Kaggle challenge, using only the tools available to you. This stage ends when you have combined the train and test data into a single dataframe successfully, to be used for downstream Data Cleaning and Feature Engineering. This is to ensure same data preprocessing techniques are applied to both partitions.",
        "data_cleaning": f"You are a Data Scientist in the Data Cleaning stage of solving a Kaggle challenge, using only the tools available to you. This stage ends when there are no missing values present in the data. This also includes the column corresponding to the target variable, that may have Nans in the test partition since the target variable is not present in the test partition. You are allowed to be innovative in filling the missing values based on your understanding of the data",
        "feature_engineering": f"You are a Data Scientist in the Feature Engineering stage of solving a Kaggle challenge, using only the tools available to you. Create new features, or delete unimportant features or transform existing features as needed. You are not allowed to delete or modify features that indicate if the row in the data belongs to the train or test partition. You are also not allowed to augment the feature corresponding to the target variable. Use your understanding of the data to aid your decisions. This stage ends when you feel that the features are good enough for modeling, and categorical and numerical features have been properly encoded. After the end of this stage, all the features should be int or float. Ensure that the number of features in the transformed data is not exorbitantly large.",
        "split_train_test": f"You are a Data Scientist in the Split Train Test stage of solving a Kaggle challenge, using only the tools available to you. Split the combined train and test data into train and test dataframes. This stage ends when the train and test dataframes are successfully split from the combined dataframe.",
        "train_data_to_features_target": f"You are a Data Scientist in the Converting the Train Data to Features and Target stage of solving a Kaggle challenge, using only the tools available to you. Convert the train data into features and target. This stage ends when the train data is successfully converted into features and target, for making downstream modeling upon.",
        "test_data_to_features": f"You are a Data Scientist in the Converting the Test Data to Features stage of solving a Kaggle challenge, using only the tools available to you. Convert the test data into features. This stage ends when the test data is successfully converted into features, for making downsream predictions upon.",
        "modeling": f"You are a Data Scientist in the Modeling stage of solving a Kaggle challenge, using only the tools available to you. Train and tune models. You might need to experiment with different model choices and properly tune your hyperparameters to get good performance. Use the provided evaluation tools to evaluate your trained models if needed. This stage ends when the agent has successfully created a model, that it considers to be the best",
        "create_submission_dataframe": f"You are a Data Scientist in the Create Submission stage of solving a Kaggle challenge, using only the tools available to you. Make predictions on the test data, and create a submission dataframe that contains the predictions in the requested format.  This stage ends when the submission dataframe in the correct format is created."
    }
    
    def plan_call(state: AgentState) -> AgentState:
        # ipdb.set_trace()
        """Create a high-level plan for the current stage of the ML pipeline."""
        current_stage = state.get("current_stage")
        if(state.get("plan", {}).get(current_stage) is not None):
            return state
        # plan_prompt = SystemMessage(content="""
        # You are a Data Scientist tasked with solving the Kaggle competition provided below. Create a high level plan of what steps you will take to solve the problem.
        # """)
        system_prompt = SystemMessage(content=stage_descriptions.get(current_stage, "You are a Data Scientist."))
        response = models.get(current_stage).invoke([f"{system_prompt}. Propose a step by step plan for the current stage of the problem i.e {current_stage}, that will help in clearing the current stage of the problem. Only propose the plan for the current stage you are at. Do not include any other stage's tasks in the plan.\n\n"] + [competition_description])
        state_dict = {**state}
        state_dict['plan'].update({current_stage: response.content})
        # return {"plan": response.content}
        return state_dict

    def model_call(state: AgentState) -> AgentState:

        # all_tools = list(tool_functions.values())
        # bind_kwargs = {'parallel_tool_calls': False} if('o3' not in os.environ['OPENAI_MODEL']) else {}
        # model = ChatOpenAI(model=os.environ['OPENAI_MODEL'],
        #                 base_url=os.environ['OPENAI_BASE_URL'],
        #                 api_key=os.environ['OPENAI_API_KEY']
        #                 ).bind_tools(
        #                     all_tools, 
        #                     **bind_kwargs
        #                 )
        current_stage = state.get("current_stage")
        # Create stage-specific system prompt with available tools
        
        system_prompt = SystemMessage(content=stage_descriptions.get(current_stage, "You are a Data Scientist."))

        # response = models.get(current_stage).invoke([system_prompt] + ['\n\nKeep the number of tool calls reasonably low. Randomly calling tools is a dumb idea. Given the context, think what tools you need to call, that would help you to solve the task\n\n'] + state["messages"])
        # return {"messages": [response]}

        response = models.get(current_stage).invoke([system_prompt] + [competition_description] + [f"\n\nThe plan to solve the current stage i.e {current_stage} of the problem is: {state.get('plan',{}).get(current_stage)}. Use the plan to guide your choice of tools to call\n\n"] + state["messages"])
        return {"messages": [response]}


    def should_continue(state: AgentState): 
        messages = state["messages"]
        last_message = messages[-1]
        
        # Check for GPT-style tool_calls
        if last_message.tool_calls: 
            return "continue"
        
        # Check for Llama JSON format in content
        if hasattr(last_message, 'content') and last_message.content:
            try:
                import json
                content = last_message.content.strip()
                if content.startswith('{') and content.endswith('}'):
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and 'name' in parsed and 'parameters' in parsed:
                        return "continue"
            except (json.JSONDecodeError, AttributeError):
                pass
        
        return "end"

    def should_advance_stage(state: AgentState):

        """Determine if we should advance to the next stage."""
        current_stage = state.get("current_stage")
        # stages = ["data_loading_eda", "data_cleaning", "feature_engineering", "modeling", "create_submission"]
        stages = ["train_data_loading", "test_data_loading", "combine_train_test", "data_cleaning", "feature_engineering", "split_train_test", "train_data_to_features_target", "test_data_to_features", "modeling", "create_submission_dataframe"]
        
        try:
            current_index = stages.index(current_stage)
            if current_index < len(stages) - 1:
                print(f"\n\n########Advancing from stage: {current_stage} to stage: {stages[current_index + 1]}########\n\n")
                return stages[current_index + 1]
            else:
                print(f"\n\n########Advancing from stage: {current_stage} to end########\n\n")
                return "end"
        except ValueError:
            raise ValueError(f"Invalid stage: {current_stage}")

    # Create the main graph
    graph = StateGraph(AgentState)
    
    # Add nodes for each stage
    for stage_name in stage_mapping.keys():
        graph.add_node(stage_name, model_call)
        graph.add_node(f"{stage_name}_plan", plan_call)
        # Create tool node for this stage (using all tools, but prompting restricts usage)
        stage_tool_node = ToolNode(tools=stage_tools[stage_name])
        graph.add_node(f"{stage_name}_tools", stage_tool_node)
        
        # Add conditional edges for this stage
        if stage_name == "create_submission":
            # Last stage goes directly to END
            graph.add_conditional_edges(
                stage_name,
                should_continue,
                {
                    "continue": f"{stage_name}_tools",
                    "end": END,
                },
            )
        else:
            # Other stages go to stage_advance
            graph.add_conditional_edges(
                stage_name,
                should_continue,
                {
                    "continue": f"{stage_name}_tools",
                    "end": "stage_advance",
                },
            )
        
        # graph.add_conditional_edges(
        #         stage_name,
        #         should_continue,
        #         {
        #             "continue": f"{stage_name}_tools",
        #             "end": "stage_advance",
        #         },
        #     )

        # Connect tools back to the stage
        graph.add_edge(f"{stage_name}_tools", stage_name)
        # Connect planner to stage
        graph.add_edge(f"{stage_name}_plan", stage_name)

    # Add stage advancement node
    # def advance_stage(state: AgentState) -> AgentState:
    #     # print("Advance stage called from:", state.get("current_stage"))
    #     next_stage = should_advance_stage(state)
    #     # print(state.get('current_stage'), '      ', next_stage)
    #     if next_stage == "end":
    #         return {"messages": state["messages"], "current_stage": "end"}
    #     else:
    #         return {"messages": state["messages"], "current_stage": next_stage}

    def advance_stage(state: AgentState) -> AgentState:
        # print("Advance stage called from:", state.get("current_stage"))
        if(state.get("current_stage") == "create_submission_dataframe"):
            return {"messages": state["messages"], "current_stage": "create_submission_dataframe"}
        else:
            next_stage = should_advance_stage(state)
            # print(state.get('current_stage'), '      ', next_stage)
            return {"messages": state["messages"], "current_stage": next_stage}

    graph.add_node("stage_advance", advance_stage)
    
    # Add conditional edges from stage_advance
    graph.add_conditional_edges(
        "stage_advance",
        # lambda state: state.get("current_stage", "end"),
        lambda state: state.get("current_stage"),
        # {
        #     "data_loading_eda": "data_loading_eda_plan",
        #     "data_cleaning": "data_cleaning_plan", 
        #     "feature_engineering": "feature_engineering_plan",
        #     "modeling": "modeling_plan",
        #     "create_submission": "create_submission_plan",
        #     # "end": END
        # },
        {
            "train_data_loading": "train_data_loading_plan",
            "test_data_loading": "test_data_loading_plan",
            "combine_train_test": "combine_train_test_plan",
            "data_cleaning": "data_cleaning_plan",
            "feature_engineering": "feature_engineering_plan",
            "split_train_test": "split_train_test_plan",
            "train_data_to_features_target": "train_data_to_features_target_plan",
            "test_data_to_features": "test_data_to_features_plan",
            "modeling": "modeling_plan",
            "create_submission_dataframe": "create_submission_dataframe_plan",
        },
    )
    
    # Set entry point
    graph.set_entry_point("train_data_loading_plan")

    app = graph.compile()

    # app.get_graph().draw_mermaid_png(output_file_path="./langgraph_imgs/hierarchical_react_ds_agent_plan.png")
    # sys.exit()

    # ------------------------------------------------------------
    # Interactive Session
    # ------------------------------------------------------------

    # human_query = """
    #     Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery. We've received a transmission from four lightyears away and things aren't looking good.

    #     The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.

    #     While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!

    #     To help rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship's damaged computer system.

    #     Help save them and change history!

    #     In this competition your task is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. To help you make these predictions, you're given a set of personal records recovered from the ship's damaged computer system.

    #     File and Data Field Descriptions
    #     train.csv - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
    #     PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
    #     HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
    #     CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
    #     Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
    #     Destination - The planet the passenger will be debarking to.
    #     Age - The age of the passenger.
    #     VIP - Whether the passenger has paid for special VIP service during the voyage.
    #     RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
    #     Name - The first and last names of the passenger.
    #     Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

    #     The train data is in the data folder and is named train.csv. 
    #     The test data is in the data folder and is named test.csv. 

    #     You have to provide me the best possible model you can for the task, using the tools provided to you. 

    #     You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

    #     In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.

    #     You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

    #     Give me the evaluation results for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves folder.

    #     Most Importantly, save your predictions for the test data, as a csv, to the submissions folder. The csv should have the following columns: PassengerId - Id for each passenger in the test set.
    #     Transported - The predicted value. For each passenger, predict either True or False. Name the file as react_agent_{model_name}.csv. Look at the sample_submission.csv file in the data folder for an example of the format the results should be in.

    # """

    # Lets go step by step, perform an operation. Describe what you are doing to me, and wait for my response before performing the next operation.


    # current_state = {"messages": [HumanMessage(content=human_query)], "current_stage": "data_loading_eda"}
    current_state = {"messages": [], "current_stage": "train_data_loading", "plan": {}}

    print("\n" + "="*80)
    print("HIERARCHICAL REACT AGENT - STAGE-BASED ML PIPELINE")
    print("="*80)
    print("Stages: train_data_loading → test_data_loading → combine_train_test → data_cleaning → feature_engineering → split_train_test → train_data_to_features_target → test_data_to_features → modeling → create_submission_dataframe")
    print("="*80)

    # Record start time
    start_time = time.time()
    print(f"Starting React DS agent execution at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    interaction_info = defaultdict(int)
    with get_openai_callback() as cb:
        response = app.invoke(current_state, config={"recursion_limit": 250})
        interaction_info['total_tokens'] += cb.total_tokens
        interaction_info['prompt_tokens'] += cb.prompt_tokens
        interaction_info['completion_tokens'] += cb.completion_tokens
        interaction_info['total_cost'] += cb.total_cost

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nExecution completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Interaction Information: {interaction_info}")


    # while True:
        
    #     user_input = input("\nYou: ")
    #     # We are resetting back to beginning. 
    #     # This is because at this point all the trajectories observed so far have been failed ones.
    #     # create_submission doesnt have access to the tools needed to address mistakes made in earlier stages.
    #     # So we are resetting back to beginning, as a hack.
    #     current_state['current_stage'] = "train_data_loading"
    #     if user_input.lower() in ['quit', 'exit', 'q']:
    #         print("Goodbye!")
    #         break
    #     if user_input.lower() in ['f']:
    #         print("Useless to interact with such a braindead agent!")
    #         break

    #     # Add the new user message to the ongoing state
    #     current_state["messages"].append(HumanMessage(content=user_input))
        
    #     # Record start time
    #     start_time = time.time()
    #     print(f"Starting hierarchical React DS agent execution at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        
    #     # Run the app
    #     try:
    #         response = app.invoke(current_state, config={"recursion_limit": 250})
    #     except Exception as e:
    #         warnings.warn(f"Error in app.invoke: {e}")
    #         response = {"messages": [HumanMessage(content="I'm sorry, I'm having trouble processing your request. Try again.")]}
        
    #     # Record end time and calculate duration
    #     end_time = time.time()
    #     total_time = end_time - start_time
        
    #     print(f"\nExecution completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    #     print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    #     # Update the full state
    #     current_state = response

    #     # Show current stage and assistant message
    #     current_stage = current_state.get("current_stage", "unknown")
    #     print(f"\nCurrent Stage: {current_stage}")
    #     print(f"\n Model ID: {mid} \n")
    #     print(f"Assistant: {response['messages'][-1].content}")