from copy import deepcopy
import random
import math
import uuid
import sys, os, json
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import argparse
import pickle

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

def save_hierarchical_lats_trajectory(multi_stage_state, search_stats, save_directory, agent_id, tool_masking):
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
                print(f"⚠ Stage '{stage_name}': No root node found, skipping trajectory collection")
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
    tool_masking_enabled = tool_masking.lower() in ['true', '1', 'yes']
    trajectory_data = {
        'agent_type': 'lats_hierarchical_parallel',
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
    tool_masking_str = "masked" if tool_masking_enabled else "unmasked"
    timestamp = int(time.time())
    trajectory_filename = f"lats_hierarchical_parallel_{feedback_str}_{tool_masking_str}_trajectory_{agent_id}_{timestamp}.json"
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

def setup_hierarchical_lats_components(competition_name: str, use_feedback: str, tool_masking: str):
    """Set up all hierarchical LATS components for a single search trajectory."""
    from dotenv import load_dotenv
    load_dotenv()

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
    
    # Check if tool masking is enabled
    tool_masking_enabled = tool_masking.lower() in ['true', '1', 'yes']
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

    # Get tool masking setting
    tool_masking_enabled = tool_masking.lower() in ['true', '1', 'yes']
    tool_masking_suffix = "masked" if tool_masking_enabled else "unmasked"
    
    if(use_feedback in ['true', 'True']):
        save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/lats_hierarchical_feedback_{tool_masking_suffix}")
    else:
        save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/lats_hierarchical_no_feedback_{tool_masking_suffix}")
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(f"model_saves/{competition_name}/", exist_ok=True)
    # sys.exit()
    
    return stage_tools, all_tools, models, expansion_chains, parser, stage_mapping, all_decorated_functions


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


def run_single_hierarchical_lats_search(trajectory_id: int, competition_name: str, use_feedback, problem_type: str, tool_masking: str, verbose: bool = False):
    """
    Run a single hierarchical LATS search for the given competition.
    
    Args:
        trajectory_id: Unique identifier for this trajectory
        competition_name: Name of the competition to solve
        verbose: Whether to print detailed progress information
    
    Returns:
        Dictionary containing search results including trajectory results and final solution
    """
    
    # Generate unique ID for this search
    search_id = f"{trajectory_id}_{uuid.uuid4()}"
    interaction_info = defaultdict(int)
    worst_case_per_query_info = defaultdict(int)

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
    
    try:
        if verbose:
            print(f"\n" + "="*80)
            print(f"STARTING HIERARCHICAL LATS SEARCH {trajectory_id}")
            print("="*80)
            print("", flush=True)
        
        # Setup hierarchical LATS components
        stage_tools, all_tools, models, expansion_chains, parser, stage_mapping, all_decorated_functions = setup_hierarchical_lats_components(competition_name, use_feedback, tool_masking)
        
        ### IN our case, this has to be a pass through function
        def generate_initial_response(state: MultiStageTreeState) -> dict:
            return state


        def try_expansion_with_reducing_context_stage(state, messages, config, expansion_chain):
            """Try expansion with progressively reducing context on errors."""
            # Calculate dynamic context sizes
            msg_len = len(messages)
            nonlocal worst_case_per_query_info
            if msg_len > 0:
                msg_lengths = [len(msg.content) if hasattr(msg, 'content') and msg.content else 0 for msg in messages]
                total_chars = sum(msg_lengths)
                avg_length = total_chars / msg_len
                max_length = max(msg_lengths)
                min_length = min(msg_lengths)
                
                if verbose:
                    print(f"\n=== MESSAGE STATISTICS ===")
                    print(f"Total messages: {msg_len}")
                    print(f"Total characters: {total_chars:,}")
                    print(f"Average message length: {avg_length:.1f} chars")
                    print(f"Longest message: {max_length:,} chars")
                    print(f"Shortest message: {min_length:,} chars")
                    print(f"===========================\n")
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
                        if verbose:
                            print(f"Attempt {i+1}: Using full context ({len(messages)} messages)")
                    else:
                        # Use last N messages
                        context_messages = messages[-context_size:] if len(messages) > context_size else messages
                        if verbose:
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
                    if verbose:
                        print(f"Success on attempt {i+1}")
                    return new_candidates, info
                    
                except Exception as e:
                    if verbose:
                        print(f"Attempt {i+1} failed: {e}")
                    if i == len(context_sizes) - 1:  # Last attempt
                        if verbose:
                            print("All attempts failed, re-raising last exception")
                        raise e
                    else:
                        if verbose:
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
            
            nonlocal all_decorated_functions, interaction_info

            current_stage = state["current_stage"]
            depth_limit = state["stage_depth_limits"][current_stage]
            
            root = state["root"]
            current_depth = root.height
            
            if verbose:
                print(f"Expanding stage '{current_stage}' - Current depth: {current_depth}/{depth_limit}")
            
            # Check if depth limit reached
            if current_depth >= depth_limit:
                if verbose:
                    print(f"Depth limit reached for stage '{current_stage}' ({current_depth}/{depth_limit})")
                return state
            if(root._is_solved and current_stage != 'create_submission_dataframe'):
                if verbose:
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
                if(info['total_tokens'] > worst_case_per_query_info['total_tokens']):
                    worst_case_per_query_info['total_tokens'] = info['total_tokens']
                    worst_case_per_query_info['prompt_tokens'] = info['prompt_tokens']
                    worst_case_per_query_info['prompt_tokens_cached'] = info['prompt_tokens_cached']
                    worst_case_per_query_info['completion_tokens'] = info['completion_tokens']
                    worst_case_per_query_info['total_cost'] = info['total_cost']
            except Exception as e:
                if verbose:
                    print(f"All expansion attempts failed for stage '{current_stage}': {e}")
                raise e
                
            # Need to make input to parser a list and take the first element
            if('llama' in os.environ['OPENAI_MODEL']):
                new_candidates = parse_llama_json({'messages': [new_candidates]})['messages'][0]
            parsed = parser.batch([new_candidates])

            if verbose:
                print('\n',parsed,'\n')

            # Get stage-specific tool node  
            current_stage_tools = stage_tools[current_stage] if current_stage in stage_tools else all_tools
            tool_node = ToolNode(current_stage_tools)

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
                        {   
                            **state,
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
            
            # If successful, use the Command to update the state
            # We only deal with scratchpad updates here
            new_node_scratchpads = defaultdict(dict)
            collected_responses = defaultdict(list)
            for i, result in tool_responses:
                # Skip if this is an error
                if isinstance(result, dict) and any(
                    isinstance(m, ToolMessage) and getattr(m, "status", None) == "error"
                    for m in result.get("messages", [])
                ):
                    collected_responses[i].append(result["messages"][0])
                    if verbose:
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
            all_candidate_rewards_dct, all_candidate_human_feedback_msgs = verifiable_rewards_stage(output_messages, traj_rewards, new_node_scratchpads, current_stage, tool_docs, problem_type=problem_type)
            all_candidate_rewards = [list(dct.values())[0] if(dct) else 0. for dct in all_candidate_rewards_dct]

            # Grow tree
            if(use_feedback in ['true', 'True']):
                child_nodes = [
                    Node_Verifiable_Rewards(cand+[feedback_msg], parent=best_candidate, reward=reward, scratchpad=new_node_scratchpads.get(i, {}))
                    for i, (cand, reward, feedback_msg) in enumerate(zip(output_messages, all_candidate_rewards, all_candidate_human_feedback_msgs))
                ]
                if verbose:
                    print(f"Using rewards with feedback")
            else:
                child_nodes = [
                    Node_Verifiable_Rewards(cand, parent=best_candidate, reward=reward, scratchpad=new_node_scratchpads.get(i, {}))
                    for i, (cand, reward) in enumerate(zip(output_messages, all_candidate_rewards))
                ]
                if verbose:
                    print(f"Using rewards without feedback")
            best_candidate.children.extend(child_nodes)
            
            # Check for solved nodes in this expansion
            solved_nodes = [node for node in child_nodes if node.is_solved]
            if solved_nodes:
                if verbose:
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
                if verbose:
                    print(f"Depth limit reached for stage '{current_stage}' ({current_depth}/{depth_limit})")
                # Check if we have solutions
                if not state["stage_solutions"][current_stage]:
                    if verbose:
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
                if verbose:
                    print(f"No solutions found for stage '{current_stage}'. Ending process.")
                return state
            
            # Check if this is the last stage
            if current_stage_idx == len(STAGE_ORDER) - 1:
                if verbose:
                    print(f"Completed final stage '{current_stage}' with {len(solutions)} solutions")
                return state
            
            # Move to next stage
            next_stage = STAGE_ORDER[current_stage_idx + 1]
            if verbose:
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
                "stage_root_nodes": state["stage_root_nodes"]
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
                # Need to find the split node corresponding to each solution train node
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
                    # Attach train solution to dummy node
                    train_copy = Node_Verifiable_Rewards(
                        messages=train_sol.messages,
                        tree_node=new_state,
                        parent=dummy_root,
                        reward=train_sol.reward,
                        scratchpad=train_sol.scratchpad
                    )
                    dummy_root.children.append(train_copy)
                    
                    # Attach test solution to train solution
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
                    # Attach test solution to dummy node
                    test_copy = Node_Verifiable_Rewards(
                        messages=test_sol.messages,
                        tree_node=new_state,
                        parent=dummy_root,
                        reward=test_sol.reward,
                        scratchpad=test_sol.scratchpad
                    )
                    dummy_root.children.append(test_copy)
                    
                    # Attach model solution to test solution
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

                if verbose:
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

            if verbose:
                print(f"Created dummy root with {len(dummy_root.children)} solution children for stage '{next_stage}'")
            
            return new_state

        def should_end(state: MultiStageTreeState):
            """Check if we should end the multi-stage process."""
            current_stage = state["current_stage"]
            current_stage_idx = STAGE_ORDER.index(current_stage)
            
            # End if we're at the last stage
            if current_stage_idx == len(STAGE_ORDER) - 1:
                if verbose:
                    print(f"Ending: Reached final stage '{current_stage}'")
                return END
            
            # Otherwise continue expanding in the new stage
            return "expand_stage"

        def generate_initial_multi_stage_response(state: MultiStageTreeState) -> MultiStageTreeState:
            """Initialize the multi-stage process."""
            if verbose:
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

        mid = search_id
        
        # Get tool masking setting
        tool_masking_enabled = tool_masking.lower() in ['true', '1', 'yes']
        tool_masking_suffix = "masked" if tool_masking_enabled else "unmasked"
        
        if(use_feedback in ['true', 'True']):
            save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/lats_hierarchical_feedback_{tool_masking_suffix}")
            question = CHALLENGES[competition_name].format(save_directory=save_directory, save_file_name=f"lats_hierarchical_feedback_{tool_masking_suffix}_agent_{mid}.csv")
        else:
            save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/lats_hierarchical_no_feedback_{tool_masking_suffix}")
            question = CHALLENGES[competition_name].format(save_directory=save_directory, save_file_name=f"lats_hierarchical_no_feedback_{tool_masking_suffix}_agent_{mid}.csv")

        start_time = time.time()
        last_step = None
        
        # Initialize multi-stage state
        multi_stage_state = initialize_multi_stage_state(question)
        
        # Run the hierarchical LATS search
        for step in graph.stream(multi_stage_state, config={"recursion_limit": 750}):
            last_step = step
            step_name, step_state = next(iter(step.items()))
            if verbose:
                print(step_name)
                print("rolled out: ", step_state["root"].height)
                print("---")

        end_time = time.time()
        total_time = end_time - start_time
        
        # Save hierarchical trajectory from all stages
        if verbose:
            print("\n" + "="*70)
            print("SAVING HIERARCHICAL LATS PARALLEL TRAJECTORY FROM ALL SUBTASKS")
            print("="*70)
        
        feedback_enabled = use_feedback in ['true', 'True']
        
        search_statistics = {
            'total_search_time': total_time,
            'total_tokens': interaction_info['total_tokens'],
            'total_cost': interaction_info['total_cost'],
            'prompt_tokens': interaction_info['prompt_tokens'], 
            'completion_tokens': interaction_info['completion_tokens'],
            'tree_height': multi_stage_state["root"].height,
            'competition_name': competition_name,
            'model_used': os.environ.get('OPENAI_MODEL', 'unknown'),
            'feedback_enabled': feedback_enabled,
            'stage_order': STAGE_ORDER[:-1],  # Exclude 'end_stage'
            'stage_depth_limits': STAGE_DEPTH_LIMITS
        }
        
        trajectory_file, trajectory_data = save_hierarchical_lats_trajectory(
            multi_stage_state=multi_stage_state,
            search_stats=search_statistics,
            save_directory=save_directory,  
            agent_id=str(mid),
            tool_masking=tool_masking
        )
        
        if verbose:
            print(f"✓ Hierarchical LATS Parallel trajectory saved to: {trajectory_file}")
            print(f"✓ Total trajectory steps across all stages: {trajectory_data['total_steps']}")
            print(f"✓ Stages completed: {trajectory_data['stages_completed']}/{trajectory_data['num_stages']}")
            print(f"✓ Tree search height: {multi_stage_state['root'].height}")
            print(f"✓ Feedback enabled: {feedback_enabled}")
            print("="*70)
        
        # Get final results from the last stage
        final_stage = STAGE_ORDER[-2]  # create_submission_dataframe is the final stage
        final_solutions = multi_stage_state["stage_solutions"][final_stage]
        
        # Only save submission if we actually found a solution in the final stage
        if final_solutions:
            # Get the best solution from the final stage
            # Select best node: (has_dataframe) * trajectory_reward
            best_node = max(final_solutions, key=lambda node: (
                int(any(hasattr(v, 'shape') and hasattr(v, 'columns') for v in node.scratchpad.values()) if node.scratchpad else False) * 
                node.get_trajectory_reward()
            ))
            if verbose:
                print(f"✓ Best solution found with reward: {best_node.get_trajectory_reward()}")
            
            # Save the submission
            if best_node.scratchpad:
                for v in best_node.scratchpad.values():
                    submission_df = v
                    # Generate unique filename with node ID and tool masking info
                    tool_masking_enabled = tool_masking.lower() in ['true', '1', 'yes']
                    tool_masking_suffix = "masked" if tool_masking_enabled else "unmasked"
                    if(use_feedback in ['true', 'True']):
                        unique_filename = os.path.join(save_directory, f'lats_hierarchical_feedback_{tool_masking_suffix}_agent_manual_{mid}.csv')
                    else:
                        unique_filename = os.path.join(save_directory, f'lats_hierarchical_no_feedback_{tool_masking_suffix}_agent_manual_{mid}.csv')
                    
                    
                    ut.save_dataframe_to_csv(submission_df, unique_filename)
                    if verbose:
                        print(f"Submission saved successfully as: {unique_filename}")
                    
                    file_exists = os.path.exists(unique_filename)
                    
                    result = {
                        'trajectory_id': trajectory_id,
                        'search_id': search_id,
                        'execution_time': end_time - start_time,
                        'best_reward': best_node.get_trajectory_reward(),
                        'final_stage': final_stage,
                        'final_solutions_count': len(final_solutions),
                        'success': file_exists,
                        'error': None if file_exists else f"Output file not found: {unique_filename}",
                        'output_file': unique_filename if file_exists else None,
                        'interaction_info': interaction_info,
                        'worst_case_per_query_info': worst_case_per_query_info,
                    }
                    break
            else:
                if verbose:
                    print("⚠ Solution found but no submission data in scratchpad")
                result = {
                    'trajectory_id': trajectory_id,
                    'search_id': search_id,
                    'execution_time': end_time - start_time,
                    'best_reward': best_node.get_trajectory_reward(),
                    'final_stage': final_stage,
                    'final_solutions_count': len(final_solutions),
                    'success': False,
                    'error': "No submission data found in best node scratchpad",
                    'output_file': None,
                    'interaction_info': interaction_info,
                    'worst_case_per_query_info': worst_case_per_query_info,
                }
        else:
            if verbose:
                print(f"⚠ No solutions found in final stage '{final_stage}' - no submission to save")
            result = {
                'trajectory_id': trajectory_id,
                'search_id': search_id,
                'execution_time': end_time - start_time,
                'best_reward': 0,
                'final_stage': final_stage,
                'final_solutions_count': 0,
                'success': False,
                'error': f"No solutions found in final stage '{final_stage}'",
                'output_file': None,
                'interaction_info': interaction_info,
                'worst_case_per_query_info': worst_case_per_query_info,
            }
        
        if verbose:
            status_symbol = "✓" if result['success'] else "⚠"
            status_msg = "found solution and created file" if result['success'] else "completed but no solution"
            print(f"{status_symbol} Hierarchical LATS Search {trajectory_id} {status_msg} in {result['execution_time']:.2f}s")
            print(f"Final stage: {result['final_stage']}, Solutions: {result['final_solutions_count']}, Best reward: {result['best_reward']}")
            if result['output_file']:
                print(f"Output file: {result['output_file']}")
            print(f"\n\n Trajctory ID: {trajectory_id}, Worst case per query information: {result['worst_case_per_query_info']}, Total Cost: {result['interaction_info']['total_cost']} \n\n")
            print(f"Interaction information: {result['interaction_info']}")
        return result
        
    except Exception as e:
        result = {
            'trajectory_id': trajectory_id,
            'search_id': search_id,
            'success': False,
            'error': str(e),
            **interaction_info
        }
        
        if verbose:
            print(f"✗ Hierarchical LATS Search {trajectory_id} failed: {e}")
        
        return result


def run_parallel_hierarchical_lats_searches(competition_name: str, num_trajectories: int, max_workers: int , use_feedback, problem_type: str, tool_masking: str, verbose: bool = True):
    """
    Run multiple hierarchical LATS searches in parallel.
    """
    
    print(f"\n" + "="*80)
    print(f"RUNNING {num_trajectories} PARALLEL HIERARCHICAL LATS SEARCHES")
    print(f"Max workers: {max_workers}")
    print(f"Competition: {competition_name}")
    print("="*80, flush=True)
    
    start_time = time.time()
    results = []
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all search trajectories
        future_to_trajectory = {
            executor.submit(run_single_hierarchical_lats_search, i, competition_name, use_feedback, problem_type, tool_masking, verbose): i 
            for i in range(num_trajectories)
        }
        
        print(f"All {num_trajectories} hierarchical LATS searches submitted to worker pool. Waiting for completion...")
        print("", flush=True)
        
        # Collect results as they complete
        completed_count = 0
        for future in as_completed(future_to_trajectory):
            trajectory_id = future_to_trajectory[future]
            completed_count += 1
            
            try:
                result = future.result()
                results.append(result)
                
                if verbose:
                    if result['success']:
                        print(f"✓ Hierarchical Search {trajectory_id} FOUND SOLUTION - reward: {result.get('best_reward', 0)} ({completed_count}/{num_trajectories})")
                    else:
                        print(f"✗ Hierarchical Search {trajectory_id} NO SOLUTION FOUND ({completed_count}/{num_trajectories})")
                    print("", flush=True)
                    
            except Exception as e:
                print(f"✗ Hierarchical Search {trajectory_id} generated an exception: {e} ({completed_count}/{num_trajectories})")
                print("", flush=True)
                results.append({
                    'trajectory_id': trajectory_id,
                    'success': False,
                    'error': str(e),
                })
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Summary statistics
    successful_searches = [r for r in results if r['success']]
    failed_searches = [r for r in results if not r['success']]
    
    print(f"\n" + "="*80)
    print("PARALLEL HIERARCHICAL LATS EXECUTION SUMMARY")
    print("="*80)
    print(f"Total searches: {num_trajectories}")
    print(f"Successful: {len(successful_searches)}")
    print(f"Failed: {len(failed_searches)}")
    print(f"Total execution time: {total_time:.2f}s")
    
    if successful_searches:
        avg_time = sum(r['execution_time'] for r in successful_searches) / len(successful_searches)
        avg_reward = sum(r['best_reward'] for r in successful_searches) / len(successful_searches)
        best_result = max(successful_searches, key=lambda x: x['best_reward'])
        
        print(f"Average search time: {avg_time:.2f}s")
        print(f"Average best reward: {avg_reward:.2f}")
        print(f"Best reward: {best_result['best_reward']} (Search {best_result['trajectory_id']})")
        print(f"Speedup factor: {avg_time * num_trajectories / total_time:.2f}x")
        
        print(f"\nBest solution found:")
        print(f"  Search ID: {best_result['trajectory_id']}")
        print(f"  Reward: {best_result['best_reward']}")
        print(f"  Final stage: {best_result['final_stage']}")
        print(f"  Solutions count: {best_result['final_solutions_count']}")
        print(f"  Output file: {best_result['output_file']}")
    
    # Print failed searches
    if failed_searches:
        print(f"\nFailed searches:")
        for result in failed_searches:
            print(f"  - Search {result['trajectory_id']}: {result['error']}")
    
    print(f"\n" + "="*80)
    print("PARALLEL HIERARCHICAL LATS EXECUTION COMPLETED")
    print("="*80)
    print("", flush=True)
    
    return results


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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run parallel hierarchical LATS searches for Kaggle competitions')
    parser.add_argument('--competition', '-c', type=str, 
                       help='Competition name (overrides COMPETITION_NAME env var)')
    parser.add_argument('--trajectories', '-t', type=int,
                       help='Number of parallel hierarchical LATS searches (overrides NUM_TRAJECTORIES env var)')
    parser.add_argument('--max_workers', '-w', type=int,
                       help='Maximum number of worker processes (overrides MAX_WORKERS env var)')
    parser.add_argument('--problem_type', '-p', type=str, choices=['binary-acc', 'binary-probs', 'regression', 'multiclass'],
                       help='Kaggle Challenge Problem type (overrides PROBLEM_TYPE env var)')
    parser.add_argument('--use_feedback', action='store_true',
                       help='Enable feedback messages (overrides USE_FEEDBACK env var)')
    parser.add_argument('--no_feedback', action='store_true',
                       help='Disable feedback messages (overrides USE_FEEDBACK env var)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', action='store_true',
                       help='Disable verbose output')
    parser.add_argument('--tool_masking', action='store_true',
                       help='Enable tool masking (use stage-specific tools)')
    parser.add_argument('--no_tool_masking', action='store_true',
                       help='Disable tool masking (use all tools for all stages)')
    
    args = parser.parse_args()
    
    # Configuration - command line args override environment variables
    NUM_TRAJECTORIES = args.trajectories if args.trajectories is not None else int(os.environ.get('NUM_TRAJECTORIES', 4))
    MAX_WORKERS = args.max_workers if args.max_workers is not None else int(os.environ.get('MAX_WORKERS', mp.cpu_count()))
    COMPETITION_NAME = args.competition if args.competition is not None else os.environ['COMPETITION_NAME']
    PROBLEM_TYPE = args.problem_type if args.problem_type is not None else os.environ['PROBLEM_TYPE']
    
    # Handle feedback flag
    if args.use_feedback:
        USE_FEEDBACK = 'true'
        feedback_source = 'CLI'
    elif args.no_feedback:
        USE_FEEDBACK = 'false'
        feedback_source = 'CLI'
    else:
        USE_FEEDBACK = os.environ.get('USE_FEEDBACK', 'false')
        feedback_source = 'ENV/DEFAULT'
    
    # Set environment variables for the worker processes
    os.environ['USE_FEEDBACK'] = USE_FEEDBACK
    
    # Handle verbose flag
    if args.quiet:
        VERBOSE = False
    elif args.verbose:
        VERBOSE = True
    else:
        VERBOSE = os.environ.get('VERBOSE', 'true').lower() == 'true'
    
    # Handle tool masking flag
    if args.tool_masking:
        TOOL_MASKING = 'true'
        tool_masking_source = 'CLI'
    elif args.no_tool_masking:
        TOOL_MASKING = 'false'
        tool_masking_source = 'CLI'
    else:
        TOOL_MASKING = os.environ.get('TOOL_MASKING', 'true')
        tool_masking_source = 'ENV/DEFAULT'
    
    # Tool masking will be passed as a parameter to worker processes
    
    # Show configuration
    print(f"Hierarchical LATS Configuration:")
    print(f"  COMPETITION_NAME: {COMPETITION_NAME} ({'CLI' if args.competition is not None else 'ENV/DEFAULT'})")
    print(f"  NUM_TRAJECTORIES: {NUM_TRAJECTORIES} ({'CLI' if args.trajectories is not None else 'ENV/DEFAULT'})")
    print(f"  MAX_WORKERS: {MAX_WORKERS} ({'CLI' if args.max_workers is not None else 'ENV/DEFAULT'})")
    print(f"  PROBLEM_TYPE: {PROBLEM_TYPE} ({'CLI' if args.problem_type is not None else 'ENV/DEFAULT'})")
    print(f"  USE_FEEDBACK: {USE_FEEDBACK} ({feedback_source})")
    print(f"  VERBOSE: {VERBOSE} ({'CLI' if args.verbose or args.quiet else 'ENV/DEFAULT'})")
    print(f"  TOOL_MASKING: {TOOL_MASKING} ({tool_masking_source})")
    print("", flush=True)

    # Run parallel hierarchical LATS searches
    results = run_parallel_hierarchical_lats_searches(
        competition_name=COMPETITION_NAME,
        num_trajectories=NUM_TRAJECTORIES,
        max_workers=MAX_WORKERS,
        use_feedback=USE_FEEDBACK,
        problem_type=PROBLEM_TYPE,
        tool_masking=TOOL_MASKING,
        verbose=VERBOSE
    )
    
    # Save results to pickle file
    feedback_str = "feedback" if USE_FEEDBACK.lower() == 'true' else "no_feedback"
    tool_masking_str = "masked" if TOOL_MASKING.lower() in ['true', '1', 'yes'] else "unmasked"
    log_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"logs/{COMPETITION_NAME}/lats_hierarchical_{feedback_str}_{tool_masking_str}")
    os.makedirs(log_directory, exist_ok=True)
    with open(os.path.join(log_directory, f"lats_hierarchical_{feedback_str}_{tool_masking_str}_{NUM_TRAJECTORIES}_results.pkl"), 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to pickle file")
    
    print(f"\nParallel hierarchical LATS execution completed with {len(results)} searches.")
