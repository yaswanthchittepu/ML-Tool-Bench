import inspect
import feature_engineering as fe
import loading as ld
import modeling as md
import hyperparameter_tuning as hpt
import evaluation as ev
import mle_utils as ut
from func_to_tool_decorator import make_get_and_set_tool, make_get_tool, make_set_tool, override_tool
import pandas as pd
import os, sys, json


# Method 1: Using inspect.getmembers to get all functions
def get_all_functions(module):
    """Get all functions from a module."""
    functions = []
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            functions.append(name)
    return functions

# Create decorator mapping based on rules
def analyze_function_signature_fe(func_name):
    """Analyze function signature to determine decorator."""
    func = getattr(fe, func_name)
    sig = inspect.signature(func)
    
    # Count DataFrame parameters
    df_params = 0
    for param_name, param in sig.parameters.items():
        if param.annotation == pd.DataFrame:
            df_params += 1
    
    # Check return type
    return_type = sig.return_annotation
    
    # Rule 1: Exactly 1 pd.DataFrame in args and return is pd.DataFrame -> override_tool
    if df_params == 1 and return_type == pd.DataFrame:
        if func_name == "get_group_aggregation":
            return "make_get_and_set_tool"
        elif func_name == "filter_dataframe":
            return "make_get_and_set_tool"
        elif func_name == "get_unique_values":
            return "make_get_and_set_tool"
        else:
            return "override_tool"
    
    # Rule 2: Exactly 1 pd.DataFrame in args and return is str -> make_get_tool
    elif df_params == 1 and return_type == str:
        return "make_get_tool"
    
    # Else: make_get_and_set_tool
    else:
        return "make_get_and_set_tool"

def analyze_function_signature_ld(func_name):
    """Analyze function signature to determine decorator for loading module."""
    if func_name == "read_data":
        return "make_set_tool"
    elif func_name == "split_data":
        return "make_get_and_set_tool"
    else:
        return None

def analyze_function_signature_md(func_name):
    return "make_get_and_set_tool"

def analyze_function_signature_hpt(func_name):
    return "make_get_and_set_tool"

def analyze_function_signature_ev(func_name):
    """Analyze function signature to determine decorator for evaluation module."""
    func = getattr(ev, func_name)
    sig = inspect.signature(func)
    return_type = sig.return_annotation
    
    if return_type == pd.DataFrame:
        return "make_get_and_set_tool"
    elif return_type == str:
        return "make_get_tool"
    else:
        return None

def analyze_function_signature_ut(func_name):
    if func_name == "save_model":
        return "make_get_tool"
    elif func_name == "load_model":
        return "make_set_tool"
    elif func_name == "save_dataframe_to_csv":
        return "make_get_tool"
    else:
        return None


if __name__ == "__main__":
    
    # module name fe.__name__
    decorator_map = {}
    tool_modules = [fe, ld, md, hpt, ev, ut]
    module_to_dec_mapper = {
        fe: analyze_function_signature_fe,
        ld: analyze_function_signature_ld,
        md: analyze_function_signature_md,
        hpt: analyze_function_signature_hpt,
        ev: analyze_function_signature_ev,
        ut: analyze_function_signature_ut,
    }

    for module in tool_modules:

        decorator_map[module.__name__] = {}
        functions = get_all_functions(module)

        # Create the mapping
        for func_name in functions:
            decorator = module_to_dec_mapper[module](func_name)
            if decorator is None:
                continue
            decorator_map[module.__name__][func_name] = decorator
    
    with open("function_to_decorator_map.json", "w") as f:
        json.dump(decorator_map, f)
    
    