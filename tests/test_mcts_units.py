"""Tree-search building blocks: nodes, UCT, stage rewards, submission check, config."""
import math

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from mltoolbench.agents.mcts.config import MCTSConfig
from mltoolbench.agents.mcts.expansion import context_sizes
from mltoolbench.agents.mcts.node import Node
from mltoolbench.agents.mcts.rewards import StageVerifier, validate_submission_frame
from mltoolbench.agents.mcts.search import Budget

# ------------------------------------------------------------------ nodes

def test_backprop_running_mean_and_path_state():
    root = Node([])
    a = Node([], root, {"x": 1}, reward=1.0)
    b = Node([], a, {"x": 2, "y": 3}, reward=0.0)
    assert root.visits == 3 and a.visits == 2 and b.visits == 1
    assert a.value == pytest.approx(0.5) and root.value == pytest.approx(1 / 3)
    assert b.path_scratchpad() == {"x": 2, "y": 3} and a.path_scratchpad() == {"x": 1}
    assert b.trajectory_reward() == 1.0 and root.height == 3


def test_uct_uses_mean_value():
    root = Node([])
    child = Node([], root, reward=1.0)
    Node([], child, reward=1.0)  # child: value 1.0 (mean), visits 2; root visits 3
    explore = math.sqrt(math.log(3) / 2)
    assert child.uct(1.0) == pytest.approx(1.0 + explore)


def test_mark_solved_ancestors():
    root = Node([])
    leaf = Node([], Node([], root))
    leaf.mark_solved(ancestors=True)
    assert root.is_solved and leaf.is_solved


# ------------------------------------------------------------------ stage rewards

def step(name, args, output):
    call = {"name": name, "args": args, "id": "c1", "type": "tool_call"}
    return [AIMessage(content="", tool_calls=[call], additional_kwargs={"tool_calls": [call]}),
            ToolMessage(output, tool_call_id="c1")]


@pytest.fixture
def shaped():
    return StageVerifier("binary-acc", "shaped")


def test_next_stage(shaped):
    assert shaped.next_stage(["init"]) == "train_data_loading"
    assert shaped.next_stage(["init", "train_data_loading", "test_data_loading"]) == "combine_train_test"
    outcome = StageVerifier("binary-acc", "outcome")
    assert outcome.next_stage(["init"]) == "modeling"
    assert outcome.next_stage(["init", "modeling", "create_submission_dataframe"]) is None


def test_loading_checks(shaped):
    ok = step("read_data", {"func_kwargs": {"filepath": "d/train.csv"}, "output": "t"}, "Applied read_data with ...")
    assert shaped.check("train_data_loading", ok, {}, "doc").reward == 1
    early = step("read_data", {"func_kwargs": {"filepath": "d/test.csv"}, "output": "t"}, "Applied read_data with ...")
    v = shaped.check("train_data_loading", early, {}, "doc")
    assert v.reward == 0 and "Penalized" in v.feedback
    hier = StageVerifier("binary-acc", "hierarchical").check("train_data_loading", early, {}, "DOC")
    assert hier.reward == 0 and "NOT loaded" in hier.feedback and hier.feedback.endswith("DOC")
    no_call = [AIMessage(content="I am done")]
    assert shaped.check("train_data_loading", no_call, {}, "").reward == 0


def test_cleaning_and_features(shaped):
    msgs = step("fillna_with_median", {}, "Applied fillna_with_median ...")
    dirty = {"all": pd.DataFrame({"a": [1.0, np.nan]})}
    assert shaped.check("data_cleaning", msgs, dirty, "").reward == 0
    assert shaped.check("data_cleaning", msgs, {"all": pd.DataFrame({"a": [1.0, 2.0]})}, "").reward == 2
    assert shaped.check("data_cleaning", msgs, {"a": 1, "b": 2}, "").reward == 0  # must touch exactly one frame
    mixed = {"all": pd.DataFrame({"a": [1, 2], "c": ["x", "y"]})}
    v = shaped.check("feature_engineering", msgs, mixed, "")
    assert v.reward == 0 and "Categorical columns found" in v.feedback
    assert shaped.check("feature_engineering", msgs, {"all": pd.DataFrame({"a": [1, 2]})}, "").reward == 2


def test_feature_target_checks(shaped):
    msgs = step("convert_dataframe_to_features_target", {}, "Applied convert_dataframe_to_features_target ...")
    assert shaped.check("train_data_to_features_target", msgs, {"X_train": 1, "Y_train": 2}, "").reward == 1
    assert shaped.check("train_data_to_features_target", msgs, {"X_test": 1}, "").reward == 0
    assert shaped.check("test_data_to_features", msgs, {"X_test": 1}, "").reward == 1


@pytest.mark.parametrize("mode,tool,cv,expected", [
    ("shaped", "tune_xgboost_classifier", 0.8, 2 + 2.5 * 0.8),
    ("outcome", "tune_xgboost_classifier", 0.8, 2.5 * 0.8),
    ("shaped", "tune_logistic_regressor", 0.5, 2 + 1.25),
    ("shaped", "tune_linear_regressor", 0.5, 2 + 1.25),
    ("shaped", "tune_xgboost_classifier", float("nan"), 2.0),
    ("outcome", "tune_xgboost_classifier", float("nan"), 0.0),
    ("hierarchical", "tune_xgboost_classifier", float("nan"), -98.0),
])
def test_modeling_reward(mode, tool, cv, expected):
    v = StageVerifier("binary-acc", mode)
    msgs = step(tool, {}, f"Applied {tool} with docstring ...")
    assert v.check("modeling", msgs, {"best_cv_score": np.float64(cv)}, "").reward == pytest.approx(expected)


def test_submission_check(shaped):
    msgs = step("predict_target", {}, "Applied predict_target ...")
    good = {"preds": pd.DataFrame({"prediction": [0, 1, 1]})}
    v = shaped.check("create_submission_dataframe", msgs, good, "")
    assert v.reward == 2 and v.solved
    assert shaped.check("create_submission_dataframe", msgs, {"preds": pd.DataFrame({"p": [0, np.nan]})}, "").reward == 0
    assert shaped.check("create_submission_dataframe", msgs, {"m": object()}, "").reward == 0


@pytest.mark.parametrize("df,problem,valid", [
    (pd.DataFrame({"id": ["a", "b"], "y": [True, False]}), "binary-acc", True),
    (pd.DataFrame({"y": [0.2, 0.9]}), "binary-probs", True),
    (pd.DataFrame({"p0": [0.8, 0.1], "p1": [0.2, 0.9]}), "binary-probs", True),
    (pd.DataFrame({"y": [0, 1]}), "binary-probs", False),            # hard labels are not probabilities
    (pd.DataFrame({"y": [1.5, 2.5]}), "regression", True),
    (pd.DataFrame({"y": [0, 3, 5]}), "multiclass", True),
    (pd.DataFrame({"a": [0], "b": [1], "c": [1]}), "binary-acc", False),
])
def test_validate_submission_frame(df, problem, valid):
    assert validate_submission_frame(df, problem)[0] is valid


def test_fe_rejects_exploding_column_count():
    wide = {"all": pd.DataFrame(np.zeros((1, 100001)))}
    msgs = step("one_hot_encode", {}, "Applied one_hot_encode")
    verdict = StageVerifier("binary-acc").check("feature_engineering", msgs, wide, "")
    assert verdict.reward == 0 and "greater than 100000." in verdict.feedback


# ------------------------------------------------------------------ config / misc

def test_config_overrides_and_rejects_unknown_options():
    assert MCTSConfig.build({"max_depth": 30}, max_depth=10).max_depth == 10
    with pytest.raises(TypeError):
        MCTSConfig.build({}, not_a_knob=1)


def test_context_sizes_schedule():
    assert context_sizes(4) == [None, 5, 1]
    assert context_sizes(25) == [None, 20, 10, 5, 1]
    assert context_sizes(30) == [None, 20, 10, 5, 1]


def test_budget():
    b = Budget(limit=3)
    assert b.take() and b.take() and not b.take()
