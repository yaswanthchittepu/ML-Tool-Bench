from copy import deepcopy
import random
import math
import uuid
import sys, os, json
import warnings

# Get the absolute path of the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import feature_engineering as fe
import hyperparameter_tuning as hpt
import evaluation as ev
import mle_utils as ut
import modeling as md
import loading as ld
from func_to_tool_decorator_lats_hierarchical import make_get_and_set_tool, make_get_tool, make_set_tool, override_tool
from challenges import CHALLENGES
# from cls import Reflection, Node, TreeState, TreeState_Verifiable_Rewards, Node_Verifiable_Rewards, verifiable_rewards
from cls_hierarchical import verifiable_rewards_stage, Node_Verifiable_Rewards, MultiStageTreeState
from langchain_core.messages import ToolMessage, AIMessage

def save_hierarchical_lats_trajectory(multi_stage_state, search_stats, save_directory, agent_id):
    """
    Save hierarchical LATS trajectory by collecting trajectories from each subtask.
    
    Args:
        multi_stage_state: The multi-stage state containing stage solutions
        search_stats: Dictionary containing search statistics
        save_directory: Directory to save the trajectory file
        agent_id: Unique identifier for this agent run
        
    Returns:
        tuple: (trajectory_file_path, trajectory_data)
    """
    import time
    import json
    import os
    
    # Collect trajectories from each stage
    stage_trajectories = {}
    total_steps = 0
    
    for stage_name in STAGE_ORDER[:-1]:  # Exclude 'end_stage'
        stage_solutions = multi_stage_state["stage_solutions"][stage_name]
        
        if stage_solutions:
            # Stage completed successfully - get best solution
            best_node = max(stage_solutions, key=lambda node: node.get_trajectory_reward())
            trajectory_messages = best_node.get_trajectory(include_reflections=False)
            node_type = "solution"
            
            print(f"✓ Stage '{stage_name}': Found {len(stage_solutions)} solutions, using best with reward {best_node.get_trajectory_reward()}")
        else:
            # Stage failed - get best terminal node from root
            def get_best_terminal_node_for_stage(root_node, target_stage):
                """Get the best terminal node that was working on the target stage"""
                all_nodes = [root_node] + root_node._get_all_children()
                
                # Filter nodes that were working on this stage
                stage_nodes = []
                for node in all_nodes:
                    # Check if this node was working on the target stage
                    # This is a heuristic - we could improve this with better stage tracking
                    if hasattr(node, 'stage') and node.stage == target_stage:
                        stage_nodes.append(node)
                    elif not hasattr(node, 'stage'):
                        # If no stage info, include all nodes as potential candidates
                        stage_nodes.append(node)
                
                if not stage_nodes:
                    stage_nodes = all_nodes
                
                # Get terminal nodes (no children or leaf nodes)
                terminal_nodes = [node for node in stage_nodes if not node.children or node.is_terminal]
                
                if not terminal_nodes:
                    terminal_nodes = stage_nodes
                
                if terminal_nodes:
                    return max(terminal_nodes, key=lambda node: node.get_trajectory_reward())
                
                return root_node
            
            if(multi_stage_state["stage_root_nodes"][stage_name] is None):
                break

            best_node = get_best_terminal_node_for_stage(multi_stage_state["stage_root_nodes"][stage_name], stage_name)
            trajectory_messages = best_node.get_trajectory(include_reflections=False)
            node_type = "best_attempt"
            
            print(f"⚠ Stage '{stage_name}': No solutions found, using best attempt with reward {best_node.get_trajectory_reward()}")
        
        # Parse trajectory messages into structured steps using your criteria
        stage_steps = []
        
        for msg in trajectory_messages:
            step_info = {}
            
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                step_info.update({
                    'step_type': 'tool_selection',
                    'action': 'selected_tools_for_execution',
                    'tools_selected': [tc['name'] for tc in msg.tool_calls],
                    'tool_calls_detail': msg.tool_calls,
                    'content': msg.content
                })
            elif hasattr(msg, 'content'):
                if(isinstance(msg, ToolMessage)):
                    step_info.update({
                        'step_type': 'tool_result',
                        'action': 'received_tool_output',
                        'content_preview': str(msg.content)[:300] + "..." if len(str(msg.content)) > 300 else str(msg.content),
                        'content_length': len(str(msg.content))
                    })
                elif isinstance(msg, AIMessage):
                    step_info.update({
                        'step_type': 'ai_reasoning',
                        'action': 'generated_ai_reasoning',
                        'content_preview': msg.content[:300] + "..." if len(msg.content) > 300 else msg.content,
                        'content_length': len(msg.content)
                    })
                else:
                    step_info.update({
                        'step_type': 'reward_feedback',
                        'action': 'generated_reward_feedback',
                        'content_preview': msg.content[:300] + "..." if len(msg.content) > 300 else msg.content,
                        'content_length': len(msg.content)
                    })
            
            if step_info:
                stage_steps.append(step_info)
        
        stage_trajectories[stage_name] = {
            'node_type': node_type,
            'node_depth': best_node.depth,
            'node_reward': best_node.get_trajectory_reward(),
            'steps': stage_steps,
            'num_steps': len(stage_steps)
        }
        
        total_steps += len(stage_steps)
    
    # Create complete trajectory data
    tool_masking_enabled = os.environ.get('TOOL_MASKING', 'true').lower() in ['true', '1', 'yes']
    trajectory_data = {
        'agent_type': 'lats_hierarchical',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'agent_id': agent_id,
        'tool_masking_enabled': tool_masking_enabled,
        'total_steps': total_steps,
        'num_stages': len([s for s in STAGE_ORDER[:-1] if multi_stage_state["stage_solutions"][s] or True]),
        'stages_completed': len([s for s in STAGE_ORDER[:-1] if multi_stage_state["stage_solutions"][s]]),
        'stage_trajectories': stage_trajectories,
        'search_statistics': search_stats
    }
    
    # Save to file
    feedback_str = "feedback" if search_stats.get('feedback_enabled', False) else "no_feedback"
    tool_masking_enabled = os.environ.get('TOOL_MASKING', 'true').lower() in ['true', '1', 'yes']
    tool_masking_str = "masked" if tool_masking_enabled else "unmasked"
    timestamp = int(time.time())
    trajectory_filename = f"lats_hierarchical_{feedback_str}_{tool_masking_str}_trajectory_{agent_id}_{timestamp}.json"
    trajectory_file = os.path.join(save_directory, trajectory_filename)
    
    with open(trajectory_file, 'w') as f:
        json.dump(trajectory_data, f, indent=2, default=str)
    
    return trajectory_file, trajectory_data


# Define stage order and depth limits
STAGE_ORDER = ['train_data_loading', 'test_data_loading', 'combine_train_test', 'data_cleaning', 'feature_engineering', 'split_train_test', 'train_data_to_features_target', 'test_data_to_features', 'modeling', 'create_submission_dataframe', 'end_stage']
STAGE_DEPTH_LIMITS = {
    "train_data_loading": 10,
    "test_data_loading": 10,
    "combine_train_test": 10,
    "data_cleaning": 15, 
    "feature_engineering": 20,
    "split_train_test": 10,
    "train_data_to_features_target": 10,
    "test_data_to_features": 10,
    "modeling": 20,
    "create_submission_dataframe": 10
}

def initialize_multi_stage_state(question: str) -> MultiStageTreeState:
    """Initialize the multi-stage tree state."""
    tree_node = {'input': question}
    root_node = Node_Verifiable_Rewards(messages=[], tree_node=tree_node)
    tree_node['root'] = root_node

    state = {
        "input": question,
        "root": root_node,
        "messages": [],
        "scratchpad": {},
        "num_tool_calls": 0,
        "current_stage": STAGE_ORDER[0],
        "stage_depth_limits": STAGE_DEPTH_LIMITS.copy(),
        "stage_solutions": {stage: [] for stage in STAGE_ORDER},
        "stage_root_nodes": {stage: None for stage in STAGE_ORDER}
    }
    state["stage_root_nodes"][STAGE_ORDER[0]] = root_node
    return state

# def get_stage_chain(stage_name: str):
#     """Get the expansion chain for a specific stage."""
#     return expansion_chains[stage_name]

from collections import deque, defaultdict
from typing import Optional, List, Dict
from typing_extensions import TypedDict
from typing import Literal
from typing import Annotated, Sequence, NotRequired, Any, Union

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage, AIMessage # The foundational class for all message types in LangGraph
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langmem.short_term import SummarizationNode, RunningSummary, summarize_messages

from langgraph.graph import END, StateGraph, START
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langgraph.prebuilt import ToolNode
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import chain as as_runnable
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableConfig
from langchain.callbacks import get_openai_callback

from pydantic import BaseModel, Field

import pandas as pd
import numpy as np
import time

import ipdb
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()


# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#     scratchpad: NotRequired[Dict[str, Any]]
#     plan: NotRequired[str]
#     summary: NotRequired[RunningSummary]
#     num_tool_calls: NotRequired[int]


if __name__ == "__main__":

    # train_df = pd.read_csv('data/spaceship_titanic/train.csv')
    # test_df = pd.read_csv('data/spaceship_titanic/test.csv')
    # print(f"Train data shape: {train_df.shape}")
    # print(f"Test data shape: {test_df.shape}")
    # sys.exit()


    allowed_functions = set(['read_data', 'create_conditional_feature', 'concatenate_train_test', 'fillna_with_median', 'fillna_with_mode',
        'extract_string_pattern', 'split_string_column', 'create_group_aggregation', 'drop_feature', 'cast_columns',
        'normalize_all_numerical_columns', 'encode_all_categorical_columns', 'split_combined_into_train_test',
        'drop_feature', 'get_features', 'tune_catboost_classifier', 'tune_random_forest_classifier',
        'tune_lightgbm_classifier', 'tune_logistic_classifier', 'tune_xgboost_classifier', 'predict_target', 
        'rename_feature', 'concatenate_dataframes'
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
            if(func_name == 'save_dataframe_to_csv'):
                continue
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


    # funcs = list(all_decorated_functions.values())
    # func = tool(funcs[0])
    # print(f" \n\n\n\n {func.name}, {func.description}, {func.args} \n\n\n\n")

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
        print(f"Stage '{stage_name}': {len(stage_tools[stage_name])} tools, Tools: {[f.name for f in stage_tools[stage_name]]}")

    # Create single model with all tools
    all_tools = list(tool_functions.values())
    bind_kwargs = {'parallel_tool_calls': False} if('o3' not in os.environ['OPENAI_MODEL']) else {}
    # model = ChatOpenAI(model=os.environ['OPENAI_MODEL'],
    #                   base_url=os.environ['OPENAI_BASE_URL'],
    #                   api_key=os.environ['OPENAI_API_KEY']
    #                 ).bind_tools(
    #                     all_tools, 
    #                     **bind_kwargs
    #                 )
    # Check if tool masking is enabled
    tool_masking_enabled = os.environ.get('TOOL_MASKING', 'true').lower() in ['true', '1', 'yes']
    print(f"🔧 TOOL MASKING: {'ENABLED - Using stage-specific tools' if tool_masking_enabled else 'DISABLED - Using all tools for all stages'}")
    
    models = {
            stage_name: ChatOpenAI(model=os.environ['OPENAI_MODEL'],
                            base_url=os.environ['OPENAI_BASE_URL'],
                            api_key=os.environ['OPENAI_API_KEY'],
                            temperature=float(os.environ['TEMPERATURE'])
                            ).bind_tools(
                                stage_tools[stage_name] if tool_masking_enabled else all_tools, 
                                **bind_kwargs
                            ) 
                        for stage_name in stage_mapping.keys()
    }

    def parse_llama_json(state):
        messages = state["messages"]
        last_message = messages[-1]

        # Check if we need to convert Llama JSON format to tool_calls
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            # Try to parse Llama JSON format
            if hasattr(last_message, 'content') and last_message.content:
                try:
                    import json
                    import string
                    import random
                    import ast
                    
                    content = last_message.content.strip()
                    
                    # Handle multiple tool calls separated by semicolons
                    if ';' in content:
                        # Split by semicolon and clean up each part
                        tool_call_strings = [part.strip() for part in content.split(';')]
                    else:
                        # Single tool call
                        tool_call_strings = [content] if content.startswith('{') and content.endswith('}') else []
                    
                    parsed_tool_calls = []
                    additional_kwargs_tool_calls = []
                    
                    for tool_call_string in tool_call_strings:
                        if tool_call_string.startswith('{') and tool_call_string.endswith('}'):
                            try:
                                # Fix malformed JSON by balancing braces
                                open_braces = tool_call_string.count('{')
                                close_braces = tool_call_string.count('}')
                                if open_braces > close_braces:
                                    # Add missing closing braces
                                    tool_call_string += '}' * (open_braces - close_braces)
                                    print(f"Fixed malformed JSON by adding {open_braces - close_braces} closing braces")
                                
                                parsed = json.loads(tool_call_string)
                                # Fix common JSON escaping issues from LLM output
                                # Replace invalid \' with valid \" in JSON strings
                                # cleaned_string = tool_call_string.replace("\\'", "'")
                                # parsed = json.loads(cleaned_string)
                                if isinstance(parsed, dict) and 'name' in parsed and 'parameters' in parsed:
                                    # Fix parameters that are string representations of Python objects
                                    parameters = parsed['parameters']
                                    
                                    for key, value in parameters.items():
                                        if isinstance(value, str):
                                            # Check if it looks like a dict/list representation
                                            if (value.startswith('{') and value.endswith('}')) or \
                                               (value.startswith('[') and value.endswith(']')) or \
                                               (value.startswith('(') and value.endswith(')')):
                                                try:
                                                    # Parse the string representation back to a Python object
                                                    parameters[key] = ast.literal_eval(value)
                                                except (ValueError, SyntaxError) as e:
                                                    print(f"Failed to parse {key} string '{value}': {e}")
                                    
                                    # Generate GPT-style call ID (call_ + 24 random characters)
                                    call_id = 'call_' + ''.join(random.choices(string.ascii_letters + string.digits, k=24))
                                    
                                    # Add to tool_calls list
                                    parsed_tool_calls.append({
                                        'name': parsed['name'],
                                        'args': parameters,
                                        'id': call_id,
                                        'type': 'tool_call'
                                    })
                                    
                                    # Add to additional_kwargs format
                                    additional_kwargs_tool_calls.append({
                                        'id': call_id,
                                        'function': {
                                            'arguments': json.dumps(parameters),
                                            'name': parsed['name']
                                        },
                                        'type': 'function'
                                    })
                            except json.JSONDecodeError as e:
                                # import ipdb; ipdb.set_trace()
                                print(f"Failed to parse individual tool call: {tool_call_string}, error: {e}")
                                continue
                    
                    # If we successfully parsed any tool calls, update the message
                    if parsed_tool_calls:
                        # Modify the existing message in-place to match GPT format
                        last_message.content = ""  # Clear content like GPT does
                        last_message.tool_calls = parsed_tool_calls
                        
                        # Ensure additional_kwargs has tool_calls for compatibility
                        if not hasattr(last_message, 'additional_kwargs'):
                            last_message.additional_kwargs = {}
                        last_message.additional_kwargs['tool_calls'] = additional_kwargs_tool_calls
                            
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"Failed to parse Llama JSON format: {e}")
        return state

    interaction_info = defaultdict(int)
    worst_case_per_query_info = defaultdict(int)
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             "You are a Data Science judge, who evaluates the goodness of tool calling trajectories to solve Machine Learning tasks on Kaggle. Reflect and grade the agents trajectory plan for the provided challenge. The trajectories should be aimed towards solving the challenge i.e generating a trained model and a valid submission file. Keep your reflections concise and to the point.",
    #         ),
    #         ("user", "{input}"),
    #         MessagesPlaceholder(variable_name="candidate"),
    #     ]
    # )

    # reflection_llm_chain = (
    #     prompt
    #     | llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(
    #         run_name="Reflection"
    #     )
    #     | PydanticToolsParser(tools=[Reflection])
    # )


    # @as_runnable
    # def reflection_chain(inputs) -> Reflection:
    #     tool_choices = reflection_llm_chain.invoke(inputs)
    #     reflection = tool_choices[0]
    #     if not isinstance(inputs["candidate"][-1], AIMessage):
    #         reflection.found_solution = False
    #     return reflection

    stage_descriptions = {
        "train_data_loading": f"You are a Data Scientist in the Data Loading stage of solving a Kaggle challenge, using only the tools available to you. This stage ends when you have loaded the train data successfully.",
        "test_data_loading": f"You are a Data Scientist in the Data Loading stage of solving a Kaggle challenge, using only the tools available to you. This stage ends when you have loaded the test data successfully.",
        "combine_train_test": f"You are a Data Scientist in the Data Loading stage of solving a Kaggle challenge, using only the tools available to you. This stage ends when you have combined the train and test data into a single dataframe successfully, to be used for downstream Data Cleaning and Feature Engineering.",
        "data_cleaning": f"You are a Data Scientist in the Data Cleaning stage of solving a Kaggle challenge, using only the tools available to you. This stage ends when there are no missing values present in the data. This also includes the column corresponding to the target variable, that may have Nans in the test partition since the target variable is not present in the test partition. You are allowed to be innovative in filling the missing values based on your understanding of the data",
        "feature_engineering": f"You are a Data Scientist in the Feature Engineering stage of solving a Kaggle challenge, using only the tools available to you. Create new features, or delete unimportant features or transform existing features as needed. You are not allowed to delete or modify features that indicate if the row in the data belongs to the train or test partition. You are also not allowed to augment the feature corresponding to the target variable. Use your understanding of the data to aid your decisions. This stage ends when the models feels that the features are good enough for modeling, and categorical and numerical features have been properly encoded. After the end of this stage, all the features should be 1) either int or float or 2) int, float, category with the number of unique values in the category columns not being exorbitantly large.",
        "split_train_test": f"You are a Data Scientist in the Split Train Test stage of solving a Kaggle challenge, using only the tools available to you. Split the combined train and test data into train and test dataframes. This stage ends when the train and test dataframes are successfully split from the combined dataframe.",
        "train_data_to_features_target": f"You are a Data Scientist in the Converting the Train Data to Features and Target stage of solving a Kaggle challenge, using only the tools available to you. Convert the train data into features and target. This stage ends when the train data is successfully converted into features and target, for making downstream modeling upon.",
        "test_data_to_features": f"You are a Data Scientist in the Converting the Test Data to Features stage of solving a Kaggle challenge, using only the tools available to you. Convert the test data into features. This stage ends when the test data is successfully converted into features, for making downsream predictions upon.",
        "modeling": f"You are a Data Scientist in the Modeling stage of solving a Kaggle challenge, using only the tools available to you. Train and tune models. You might need to experiment with different model choices and properly tune your hyperparameters to get good performance. Use the provided evaluation tools to evaluate your trained models if needed. This stage ends when the agent has successfully created a model, that it considers to be the best",
        "create_submission_dataframe": f"You are a Data Scientist in the Create Submission stage of solving a Kaggle challenge, using only the tools available to you. Make predictions on the test data, and create a submission dataframe that contains the predictions in the requested format.  This stage ends when the submission dataframe in the correct format is created."
    }

    prompt_templates = {}
    initial_answer_chains = {}
    expansion_chains = {}
    for stage_name in stage_mapping.keys():
        prompt_templates[stage_name] = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{stage_descriptions[stage_name]}. Propose tool candidates that would help solving the problem at the current stage. Your Job is to only call tool candidates at the current step.",
                    # Any interaction apart from this is not allowed. You are allowed to propose at most 5 tool candidates. You can also propose less than 5 or even a single tool candidate if you think there are no other valid options at this stage.
                    # Also provide the rationale for your choice of tool candidates. You are allowed to propose at most 5 tool candidates. You can also propose less than 5 or even a single tool candidate if you think there are no other valid options at this stage.
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="messages", optional=True),
            ]
        )

        initial_answer_chains[stage_name] = prompt_templates[stage_name] | models[stage_name].with_config(
            run_name="GenerateInitialCandidate"
        )

        expansion_chains[stage_name] = prompt_templates[stage_name] | models[stage_name].with_config(
            run_name="GenerateCandidates"
        )


    parser = JsonOutputToolsParser(return_id=True)

    # initial_response = initial_answer_chain.invoke(
    #     {"input": "Write a research report on lithium pollution."}
    # )

    ### IN our case, this has to be a pass through function
    def generate_initial_response(state: MultiStageTreeState) -> dict:
        return state


    def generate_candidates(messages: ChatPromptValue, config: RunnableConfig):
        n = config["configurable"].get("N", 5)
        bound_kwargs = llm.bind_tools(tools=tools).kwargs
        # chat_result = llm.generate(
        #     [messages.to_messages()],
        #     n=n,
        #     callbacks=config["callbacks"],
        #     run_name="GenerateCandidates",
        #     **bound_kwargs,
        # )
        chat_result = llm.generate(
            [messages.to_messages()],
            n=1,
            callbacks=config["callbacks"],
            run_name="GenerateCandidates",
            **bound_kwargs,
        )
        return [gen.message for gen in chat_result.generations[0]]


    # expansion_chain = prompt_template | generate_candidates

    # expansion_chain = prompt_template | llm.bind_tools(tools=tools).with_config(
    #     run_name="GenerateCandidates"
    # )

    # res = expansion_chain.invoke({"input": "Write a research report on lithium pollution."})
    # print(res)

    def try_expansion_with_reducing_context_stage(state, messages, config, expansion_chain):
        """Try expansion with progressively reducing context on errors."""
        # context_sizes = [None, 20, 15, 10, 5]  # None means full context
        # Calculate dynamic context sizes
        msg_len = len(messages)
        global worst_case_per_query_info
        if msg_len > 0:
            msg_lengths = [len(msg.content) if hasattr(msg, 'content') and msg.content else 0 for msg in messages]
            total_chars = sum(msg_lengths)
            avg_length = total_chars / msg_len
            max_length = max(msg_lengths)
            min_length = min(msg_lengths)
            
            # print(f"\n=== MESSAGE STATISTICS ===")
            # print(f"Total messages: {msg_len}")
            # print(f"Total characters: {total_chars:,}")
            # print(f"Average message length: {avg_length:.1f} chars")
            # print(f"Longest message: {max_length:,} chars")
            # print(f"Shortest message: {min_length:,} chars")
            # print(f"===========================\n")
        if msg_len <= 10:
            context_sizes = [None, 5, 1]  # Just use full context if <= 10 messages
        else:
            # Start with nearest multiple of 10 less than msg_len
            start_size = (msg_len // 10) * 10
            if start_size == msg_len:
                start_size = msg_len - 10
            
            # Generate decreasing sizes by 10, minimum 10
            context_sizes = [None]  # Start with full context
            current_size = start_size
            while current_size >= 10:
                context_sizes.append(current_size)
                current_size -= 10
            context_sizes.extend([5, 1])
        info = defaultdict(int)
        for i, context_size in enumerate(context_sizes):
            try:
                if context_size is None:
                    # Use full context
                    context_messages = messages
                    print(f"Attempt {i+1}: Using full context ({len(messages)} messages)")
                else:
                    # Use last N messages
                    context_messages = messages[-context_size:] if len(messages) > context_size else messages
                    print(f"Attempt {i+1}: Using last {context_size} messages ({len(context_messages)} messages)")
                
                
                with get_openai_callback() as cb:
                    new_candidates = expansion_chain.invoke(
                        {"input": state["input"], "messages": context_messages}, config
                    )
                    info['total_tokens'] += cb.total_tokens
                    info['prompt_tokens'] += cb.prompt_tokens
                    info['completion_tokens'] += cb.completion_tokens
                    info['total_cost'] += cb.total_cost
                    if(cb.total_tokens > worst_case_per_query_info['total_tokens']):
                        worst_case_per_query_info['total_tokens'] = cb.total_tokens
                        worst_case_per_query_info['prompt_tokens'] = cb.prompt_tokens
                        worst_case_per_query_info['prompt_tokens_cached'] = cb.prompt_tokens_cached
                        worst_case_per_query_info['completion_tokens'] = cb.completion_tokens
                        worst_case_per_query_info['total_cost'] = cb.total_cost
                
                print(f"Success on attempt {i+1}")
                return new_candidates, info
                
            except Exception as e:
                # import ipdb; ipdb.set_trace()
                print(f"Attempt {i+1} failed: {e}")
                if i == len(context_sizes) - 1:  # Last attempt
                    print("All attempts failed, re-raising last exception")
                    raise e
                else:
                    print(f"Retrying with smaller context...")
                    continue

    # This is the select function for the tree
    def select(root: Node_Verifiable_Rewards) -> dict:
        """Starting from the root node a child node is selected at each tree level until a leaf node is reached."""

        if not root.children:
            return root

        node = root
        while node.children:
            max_child = max(node.children, key=lambda child: child.upper_confidence_bound(exploration_weight=5.0))
            node = max_child

        return node


    def expand_stage(state: MultiStageTreeState, config: RunnableConfig) -> dict:
        """Expand the tree for the current stage within depth limit."""
        
        # import ipdb; ipdb.set_trace()

        global all_decorated_functions, interaction_info

        current_stage = state["current_stage"]

        # if(STAGE_ORDER.index(current_stage) >= STAGE_ORDER.index('create_submission_dataframe')):
        #     import ipdb; ipdb.set_trace()

        depth_limit = state["stage_depth_limits"][current_stage]
        
        root = state["root"]
        current_depth = root.height
        
        print(f"Expanding stage '{current_stage}' - Current depth: {current_depth}/{depth_limit}")
        
        # Check if depth limit reached
        if current_depth >= depth_limit:
            print(f"Depth limit reached for stage '{current_stage}' ({current_depth}/{depth_limit})")
            return state
        if(root._is_solved and current_stage != 'create_submission_dataframe'):
            print(f"Root is already solved for stage '{current_stage}'. Skipping expansion.")
            return state
            
        best_candidate: Node_Verifiable_Rewards = select(root)
        # Combine all messages from root to best_candidate
        messages = best_candidate.get_trajectory()
        # Combine all scratchpads from root to best_candidate
        scratchpad = best_candidate.get_trajectory_scratchpad()
        # Accrue rewards along trajectory
        traj_rewards = best_candidate.get_trajectory_reward()
        
        # Get stage-specific expansion chain
        expansion_chain = expansion_chains[current_stage]
        
        # Generate candidates using stage-specific chain
        try:
            new_candidates, info = try_expansion_with_reducing_context_stage(state, messages, config, expansion_chain)
            interaction_info['total_tokens'] += info['total_tokens']
            interaction_info['prompt_tokens'] += info['prompt_tokens']
            interaction_info['completion_tokens'] += info['completion_tokens']
            interaction_info['total_cost'] += info['total_cost']
        except Exception as e:
            print(f"All expansion attempts failed for stage '{current_stage}': {e}")
            raise e
        # parsed = parser.batch(new_candidates)
        # Need to make input to parser a list and take the first element
        if('llama' in os.environ['OPENAI_MODEL']):
            new_candidates = parse_llama_json({'messages': [new_candidates]})['messages'][0]
        parsed = parser.batch([new_candidates])

        print('\n',parsed,'\n')

        # Get stage-specific tool node  
        current_stage_tools = stage_tools[current_stage] if current_stage in stage_tools else all_tools
        tool_node = ToolNode(current_stage_tools)

        # YC: need to change how we id tool calls
        flattened = [
            (i, tool_call)
            for i, tool_call in enumerate(parsed[0])
        ]

        tool_docs = [
            all_decorated_functions[tool_call["type"]].__doc__ if tool_call.get("type", None) is not None and tool_call["type"] in all_decorated_functions else ""
            for i, tool_call in flattened
        ]
        tool_responses = [
            (
                i,
                tool_node.invoke(
                    # Change the func_to_tool decarotor to only return the new objects in the scratchpad (shouldnt have the old ones)
                    {   
                        **state,
                        # We override the scratchpad and messages here
                        # scrathpad is cumulative from root to best_candidate
                        # scratchpad is used by the tool to access objects (Eg: dataframes, models, etc in our case)
                        # messages is just an AI message with tool info and args
                        "scratchpad": scratchpad,
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": tool_call["type"],
                                        "args": tool_call["args"],
                                        "id": tool_call["id"],
                                    }
                                ],
                            )
                        ]
                    }
                ),
            )
            for i, tool_call in flattened
        ]
        print(f"Tool responses: {tool_responses}")
        # If successful, use the Commmand to update the state
        # We only deal with scratchpad updates here
        # All the following updates can be merged in a single loop for efficiency ## Future work
        new_node_scratchpads = defaultdict(dict)
        collected_responses = defaultdict(list)
        
        for i, result in tool_responses:
            # Skip if this is an error
            if isinstance(result, dict) and any(
                isinstance(m, ToolMessage) and getattr(m, "status", None) == "error"
                for m in result.get("messages", [])
            ):
                collected_responses[i].append(result["messages"][0])
                print(f"Tool call {i} failed, skipping.")
                continue

            # If it's a Command, apply the update
            if isinstance(result, list) and all(isinstance(res, Command) for res in result):
                update = result[0].update
                if "scratchpad" in update:
                    new_node_scratchpads[i] = update["scratchpad"]
                if "messages" in update:
                    collected_responses[i].append(update["messages"][0])
            else:
                warnings.warn(f"Unexpected result type for tool call {i}: {type(result)} \n\n Result: {result}")

        output_messages = []

        def split_ai_message_by_tool_calls(ai_message: AIMessage) -> list[AIMessage]:
            tool_calls = ai_message.tool_calls
            if not tool_calls:
                return [ai_message]  # No tool calls, return original message

            split_messages = []
            for tool_call in tool_calls:
                new_message = AIMessage(
                    content=ai_message.content,
                    tool_calls=[tool_call],
                    additional_kwargs={"tool_calls": [tool_call]},
                    response_metadata=ai_message.response_metadata,
                )
                split_messages.append(new_message)
            return split_messages

        new_candidates_split = split_ai_message_by_tool_calls(new_candidates)
        # since new_candidates is not a list, I was seeing the candidates as just 'content' and not a Message class. So fixing this
        for i, candidate in enumerate(new_candidates_split):
            output_messages.append([candidate] + collected_responses[i])

        # Define our own verifiable rewards here
        # output message is a list of AIMessages, ToolMessages
        # contains info about what tool was called with what args, and then the response from the tool
        all_candidate_rewards_dct, all_candidate_human_feedback_msgs = verifiable_rewards_stage(output_messages, traj_rewards, new_node_scratchpads, current_stage, tool_docs, problem_type=os.environ['PROBLEM_TYPE'])
        all_candidate_rewards = [list(dct.values())[0] if(dct) else 0. for dct in all_candidate_rewards_dct]

        # Grow tree
        if(os.environ['USE_FEEDBACK'] in ['true', 'True']):
            child_nodes = [
                Node_Verifiable_Rewards(cand+[feedback_msg], parent=best_candidate, reward=reward, scratchpad=new_node_scratchpads.get(i, {}))
                for i, (cand, reward, feedback_msg) in enumerate(zip(output_messages, all_candidate_rewards, all_candidate_human_feedback_msgs))
            ]
            print(f"Using rewards with feedback")
        else:
            child_nodes = [
                Node_Verifiable_Rewards(cand, parent=best_candidate, reward=reward, scratchpad=new_node_scratchpads.get(i, {}))
                for i, (cand, reward) in enumerate(zip(output_messages, all_candidate_rewards))
            ]
            print(f"Using rewards without feedback")
        best_candidate.children.extend(child_nodes)
        
        # Check for solved nodes in this expansion
        solved_nodes = [node for node in child_nodes if node.is_solved]
        if solved_nodes:
            print(f"Found {len(solved_nodes)} solved nodes in stage '{current_stage}'")
            state["stage_solutions"][current_stage].extend(solved_nodes)
        
        return state

    def should_continue_stage(state: MultiStageTreeState):
        """Determine whether to continue expansion in current stage or transition."""
        current_stage = state["current_stage"]
        depth_limit = state["stage_depth_limits"][current_stage]
        current_depth = state["root"].height

        # Check if depth limit reached for current stage
        if current_depth >= depth_limit:
            print(f"Depth limit reached for stage '{current_stage}' ({current_depth}/{depth_limit})")
            # Check if we have solutions
            if not state["stage_solutions"][current_stage]:
                print(f"No solutions found in stage '{current_stage}'. Ending.")
                return END
            return "transition_stage"
        
        # Continue expanding if depth limit not reached
        return "expand_stage"

    def transition_stage(state: MultiStageTreeState) -> MultiStageTreeState:
        """Transition to the next stage or end if no solutions."""
        current_stage = state["current_stage"]
        current_stage_idx = STAGE_ORDER.index(current_stage)
        
        # Check if we have solutions for current stage
        solutions = state["stage_solutions"][current_stage]
        if not solutions:
            print(f"No solutions found for stage '{current_stage}'. Ending process.")
            return state
        
        # Check if this is the last stage
        if current_stage_idx == len(STAGE_ORDER) - 1:
            print(f"Completed final stage '{current_stage}' with {len(solutions)} solutions")
            return state
        
        # Move to next stage
        next_stage = STAGE_ORDER[current_stage_idx + 1]
        print(f"Transitioning from '{current_stage}' to '{next_stage}' with {len(solutions)} solutions")
        
        # Create complete state for next stage
        new_state = {
            "input": state["input"],
            "root": None,  # Will be set after creating dummy_root
            "messages": [],
            "scratchpad": {},
            "num_tool_calls": 0,
            "current_stage": next_stage,
            "stage_depth_limits": state["stage_depth_limits"],
            "stage_solutions": state["stage_solutions"],
            "stage_root_nodes": state["stage_root_nodes"],
        }
        
        # Create dummy root that references the complete new_state
        dummy_root = Node_Verifiable_Rewards(messages=[], tree_node=new_state)
        new_state["root"] = dummy_root
        new_state["stage_root_nodes"][next_stage] = dummy_root
        next_stage_children_max_limit = 10
        

        if(current_stage in ['data_cleaning', 'feature_engineering', 'split_train_test']):
            # Add all solutions as children of the dummy root
            random.shuffle(solutions)
            solutions = solutions[:next_stage_children_max_limit]
            for solution in solutions:
                # Create a copy of the solution node but reset its parent to dummy_root
                solution_copy = Node_Verifiable_Rewards(
                    messages=solution.messages,
                    tree_node=new_state,
                    parent=dummy_root,
                    reward=solution.reward,
                    scratchpad=solution.scratchpad
                )
                dummy_root.children.append(solution_copy)
        elif(current_stage == 'train_data_to_features_target'):
            # Need to find the split node vcorresponding to each solution train node
            random.shuffle(solutions)
            solutions = solutions[:next_stage_children_max_limit]
            split_solutions = []
            for solution in solutions:
                split_sol = solution
                while(split_sol.parent.parent):
                    split_sol = split_sol.parent
                split_solutions.append(split_sol)
            
            for solution, split_solution in zip(solutions, split_solutions):
                # Create copy of split_solution as child of dummy root
                split_copy = Node_Verifiable_Rewards(
                    messages=split_solution.messages,
                    tree_node=new_state,
                    parent=dummy_root,
                    reward=split_solution.reward,
                    scratchpad=split_solution.scratchpad
                )
                dummy_root.children.append(split_copy)
                
                # Create copy of modeling solution as child of split_solution
                sol_copy = Node_Verifiable_Rewards(
                    messages=solution.messages,
                    tree_node=new_state,
                    parent=split_copy,
                    reward=solution.reward,
                    scratchpad=solution.scratchpad
                )
                split_copy.children.append(sol_copy)
        elif(current_stage == 'test_data_to_features'):
            # get train solutions first
            random.shuffle(solutions)
            solutions = solutions[:next_stage_children_max_limit]
            train_solutions = []
            for solution in solutions:
                train_sol = solution
                while(train_sol.parent.parent.parent):
                    train_sol = train_sol.parent
                train_solutions.append(train_sol)
            for train_sol, test_sol in zip(train_solutions, solutions):
                # Attatch train solution to dummy node
                train_copy = Node_Verifiable_Rewards(
                    messages=train_sol.messages,
                    tree_node=new_state,
                    parent=dummy_root,
                    reward=train_sol.reward,
                    scratchpad=train_sol.scratchpad
                )
                dummy_root.children.append(train_copy)
                
                # Attatch test solution to train solution
                test_copy = Node_Verifiable_Rewards(
                    messages=test_sol.messages,
                    tree_node=new_state,
                    parent=train_copy,
                    reward=test_sol.reward,
                    scratchpad=test_sol.scratchpad
                )
                train_copy.children.append(test_copy)
        elif(current_stage == 'modeling'):
            random.shuffle(solutions)
            solutions = solutions[:next_stage_children_max_limit]
            test_solutions = []
            for solution in solutions:
                test_sol = solution
                while(test_sol.parent.parent.parent):
                    test_sol = test_sol.parent
                test_solutions.append(test_sol)
            
            for test_sol, model_sol in zip(test_solutions, solutions):
                # Attatch test solution to dummy node
                test_copy = Node_Verifiable_Rewards(
                    messages=test_sol.messages,
                    tree_node=new_state,
                    parent=dummy_root,
                    reward=test_sol.reward,
                    scratchpad=test_sol.scratchpad
                )
                dummy_root.children.append(test_copy)
                
                # Attatch model solution to test solution
                model_copy = Node_Verifiable_Rewards(
                    messages=model_sol.messages,
                    tree_node=new_state,
                    parent=test_copy,
                    reward=model_sol.reward,
                    scratchpad=model_sol.scratchpad
                )
                test_copy.children.append(model_copy)
        elif(current_stage == 'test_data_loading'):
            # Add the best solutions of both the 'train_data_loading' and 'test_data_loading' stages sequentially
            # Get best solution from train_data_loading
            train_solutions = state["stage_solutions"]["train_data_loading"]
            best_train_solution = max(train_solutions, key=lambda node: node.reward)
            
            # Create copy of best train solution as child of dummy root
            train_copy = Node_Verifiable_Rewards(
                messages=best_train_solution.messages,
                tree_node=new_state,
                parent=dummy_root,
                reward=best_train_solution.reward,
                scratchpad=best_train_solution.scratchpad
            )
            dummy_root.children.append(train_copy)
            
            # Get best solution from current stage (test_data_loading)
            best_test_solution = max(solutions, key=lambda node: node.reward)
            
            # Create copy of best test solution as child of train solution
            test_copy = Node_Verifiable_Rewards(
                messages=best_test_solution.messages,
                tree_node=new_state,
                parent=train_copy,
                reward=best_test_solution.reward,
                scratchpad=best_test_solution.scratchpad
            )
            train_copy.children.append(test_copy)

            print(f"Created dummy node and added best train and test solutions as descendants. test solution is a child of train solution. Moving to next stage '{next_stage}'")
        else:
            # Only add the best solution as the child of the dummy root
            best_solution = max(solutions, key=lambda node: node.reward)
            solution_copy = Node_Verifiable_Rewards(
                    messages=best_solution.messages,
                    tree_node=new_state,
                    parent=dummy_root,
                    reward=best_solution.reward,
                    scratchpad=best_solution.scratchpad
                )
            dummy_root.children.append(solution_copy)

        print(f"Created dummy root with {len(dummy_root.children)} solution children for stage '{next_stage}'")
        
        return new_state

    def should_end(state: MultiStageTreeState):
        """Check if we should end the multi-stage process."""
        current_stage = state["current_stage"]
        current_stage_idx = STAGE_ORDER.index(current_stage)
        
        # End if we're at the last stage
        if current_stage_idx == len(STAGE_ORDER) - 1:
            print(f"Ending: Reached final stage '{current_stage}'")
            return END
        
        # Otherwise continue expanding in the new stage
        return "expand_stage"


    def generate_initial_multi_stage_response(state: MultiStageTreeState) -> MultiStageTreeState:
        """Initialize the multi-stage process."""
        print(f"Starting multi-stage LATS with stages: {STAGE_ORDER}")
        print(f"Stage depth limits: {STAGE_DEPTH_LIMITS}")
        return state

    builder = StateGraph(MultiStageTreeState)
    builder.add_node("start", generate_initial_multi_stage_response)
    builder.add_node("expand_stage", expand_stage)
    builder.add_node("transition_stage", transition_stage)
    builder.add_edge(START, "start")

    builder.add_conditional_edges(
        "start",
        # Start with first stage expansion
        lambda state: "expand_stage",
        ["expand_stage"],
    )
    
    builder.add_conditional_edges(
        "expand_stage",
        # Either continue expanding, transition, or end
        should_continue_stage,
        ["expand_stage", "transition_stage", END],
    )
    
    builder.add_conditional_edges(
        "transition_stage",
        # Either end or continue with next stage
        should_end,
        ["expand_stage", END],
    )

    graph = builder.compile()

    mid = uuid.uuid4()
    
    # Get tool masking setting
    tool_masking_enabled = os.environ.get('TOOL_MASKING', 'true').lower() in ['true', '1', 'yes']
    tool_masking_suffix = "masked" if tool_masking_enabled else "unmasked"
    
    if(os.environ['USE_FEEDBACK'] in ['true', 'True']):
        save_directory = f"submissions/{os.environ['COMPETITION_NAME']}/lats_hierarchical_feedback_{tool_masking_suffix}"
        question = CHALLENGES[os.environ['COMPETITION_NAME']].format(save_directory=save_directory, save_file_name=f"lats_hierarchical_feedback_{tool_masking_suffix}_agent_{mid}.csv")
    else:
        save_directory = f"submissions/{os.environ['COMPETITION_NAME']}/lats_hierarchical_no_feedback_{tool_masking_suffix}"
        question = CHALLENGES[os.environ['COMPETITION_NAME']].format(save_directory=save_directory, save_file_name=f"lats_hierarchical_no_feedback_{tool_masking_suffix}_agent_{mid}.csv")
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(f"model_saves/{os.environ['COMPETITION_NAME']}/", exist_ok=True)
    # ipdb.set_trace()

    # In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.

    # You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

    # Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves folder.

    # Most Importantly, save your predictions for the test data, as a csv, to the submissions folder. The csv should have the following columns: PassengerId - Id for each passenger in the test set.
    # Transported - The predicted value. For each passenger, predict either True or False. Name the file as react_agent_{model_name}.csv. Look at the sample_submission.csv file in the data folder for an example of the format the results should be in.

    last_step = None
    
    # Initialize multi-stage state
    multi_stage_state = initialize_multi_stage_state(question)
    
    # Record start time
    start_time = time.time()
    print(f"Starting LATS hierarchical search at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    # for step in graph.stream({"input": question}):
    for step in graph.stream(multi_stage_state, config={"recursion_limit": 750}):
        last_step = step
        step_name, step_state = next(iter(step.items()))
        print(step_name)
        print("rolled out: ", step_state["root"].height)
        print("---")
    
    # Record end time and calculate duration
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nSearch completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total search time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Interaction info: {interaction_info}")
    print(f"Worst case per query info: {worst_case_per_query_info}")
    
    # Save hierarchical trajectory from all stages
    print("\n" + "="*70)
    print("SAVING HIERARCHICAL LATS TRAJECTORY FROM ALL SUBTASKS")
    print("="*70)
    
    feedback_enabled = os.environ.get('USE_FEEDBACK') in ['true', 'True']
    
    search_statistics = {
        'total_search_time': total_time,
        'total_tokens': interaction_info['total_tokens'],
        'total_cost': interaction_info['total_cost'],
        'prompt_tokens': interaction_info['prompt_tokens'], 
        'completion_tokens': interaction_info['completion_tokens'],
        'tree_height': multi_stage_state["root"].height,
        'competition_name': os.environ.get('COMPETITION_NAME', 'unknown'),
        'model_used': os.environ.get('OPENAI_MODEL', 'unknown'),
        'feedback_enabled': feedback_enabled,
        'stage_order': STAGE_ORDER[:-1],  # Exclude 'end_stage'
        'stage_depth_limits': STAGE_DEPTH_LIMITS
    }
    
    trajectory_file, trajectory_data = save_hierarchical_lats_trajectory(
        multi_stage_state=multi_stage_state,
        search_stats=search_statistics,
        save_directory=save_directory,  
        agent_id=str(mid)
    )
    
    print(f"✓ Hierarchical LATS trajectory saved to: {trajectory_file}")
    print(f"✓ Total trajectory steps across all stages: {trajectory_data['total_steps']}")
    print(f"✓ Stages completed: {trajectory_data['stages_completed']}/{trajectory_data['num_stages']}")
    print(f"✓ Tree search height: {multi_stage_state['root'].height}")
    print(f"✓ Feedback enabled: {feedback_enabled}")
    
    # Get final results from the last stage
    final_stage = STAGE_ORDER[-2]
    final_solutions = multi_stage_state["stage_solutions"][final_stage]
    
    # Only save submission if we actually found a solution in the final stage
    if final_solutions:
        # Get the best solution from the final stage
        # Select best node: (has_dataframe) * trajectory_reward  
        best_node = max(final_solutions, key=lambda node: (
            int(any(hasattr(v, 'shape') and hasattr(v, 'columns') for v in node.scratchpad.values()) if node.scratchpad else False) * 
            node.get_trajectory_reward()
        ))
        print(f"✓ Best solution found with reward: {best_node.get_trajectory_reward()}")
        
        # Save the submission
        if best_node.scratchpad:
            submission_df = list(best_node.scratchpad.values())[0]
            # Generate unique filename with node ID and tool masking info
            if(os.environ['USE_FEEDBACK'] in ['true', 'True']):
                unique_filename = os.path.join(save_directory, f'lats_hierarchical_feedback_{tool_masking_suffix}_agent_manual_{mid}.csv')
            else:
                unique_filename = os.path.join(save_directory, f'lats_hierarchical_no_feedback_{tool_masking_suffix}_agent_manual_{mid}.csv')
            ut.save_dataframe_to_csv(submission_df, unique_filename)
            print(f"✓ Submission saved successfully as: {unique_filename}")
        else:
            print("⚠ Solution found but no submission data in scratchpad")
    else:
        print(f"⚠ No solutions found in final stage '{final_stage}' - no submission to save")
    
    print("="*70)
        
        # Try to get solutions from earlier stages
        # for stage in reversed(STAGE_ORDER[:-1]):
        #     if multi_stage_state["stage_solutions"][stage]:
        #         print(f"Found solutions in earlier stage '{stage}': {len(multi_stage_state['stage_solutions'][stage])}")
        #         break
    # best_trajectory = best_node.get_trajectory(include_reflections=False)
    # print("Best solved trajectory:")
    # for msg in best_trajectory:
    #     print(f"  {msg.content}")

    # def find_best_solved_leaf(node):
    #     if not node.children:  # This is a leaf node
    #         return node, sum(node.reward.values())-0.1
    #     best_leaf, best_leaf_reward = None, float('-inf')
    #     for child in node.children:
    #         if(child.is_solved):
    #             leaf, leaf_reward = find_best_solved_leaf(child)
    #             if(leaf_reward > best_leaf_reward):
    #                 best_leaf, best_leaf_reward = leaf, leaf_reward
    #     return best_leaf, best_leaf_reward+sum(node.reward.values())-0.1
    
    # best_solved_leaf, best_reward = find_best_solved_leaf(root_node)

    # if best_solved_leaf:
    #     print(f"Found best solved leaf with cumulative reward: {best_reward}")
    #     best_trajectory = best_solved_leaf.get_trajectory(include_reflections=False)
    #     print("Best solved trajectory:")
    #     for msg in best_trajectory:
    #         print(f"  {msg.content}")
    # else:
    #     print("No solved leaf nodes found")

    # solution_node = last_step["expand"]["root"].get_best_solution()
    # best_trajectory = solution_node.get_trajectory(include_reflections=False)
    # print(best_trajectory[-1].content)
    # question = "Generate a table with the average size and weight, as well as the oldest recorded instance for each of the top 5 most common birds."
    # last_step = None
    # for step in graph.stream({"input": question}):
    #     last_step = step
    #     step_name, step_state = next(iter(step.items()))
    #     print(step_name)
    #     print("rolled out: ", step_state["root"].height)
    #     print("---")

    # solution_node = last_step["expand"]["root"].get_best_solution()
    # best_trajectory = solution_node.get_trajectory(include_reflections=False)
    # print(best_trajectory[-1].content)

    # question = "Write out magnus carlson series of moves in his game against Alireza Firouzja and propose an alternate strategy"
    # last_step = None
    # for step in graph.stream({"input": question}):
    #     last_step = step
    #     step_name, step_state = next(iter(step.items()))
    #     print(step_name)
    #     print("rolled out: ", step_state["root"].height)
    #     print("---")

    # solution_node = last_step["expand"]["root"].get_best_solution()
    # best_trajectory = solution_node.get_trajectory(include_reflections=False)
    # print(best_trajectory[-1].content)