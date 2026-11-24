"""End-to-end runs of the tree-search baselines with a scripted LLM, graded by the runner."""
import pytest
from fakes import ScriptedChatModel, scripted_config, tool_call
from langchain_core.messages import AIMessage

from mltoolbench.agents.registry import build_agent
from mltoolbench.runner import run_one


def mcts_pipeline(task):
    """The 10-stage pipeline (MCTS agents have no save tool: the framework writes the submission)."""
    pub = task.public_dir
    return [
        tool_call("read_data", output="train", func_kwargs={"filepath": str(pub / "train.csv")}),
        tool_call("read_data", output="test", func_kwargs={"filepath": str(pub / "test.csv")}),
        tool_call("concatenate_train_test", bindings={"train_df": "train", "test_df": "test"}, output="all"),
        tool_call("fillna_with_median", bindings={"df": "all"}, func_kwargs={"columns": ["Age"]}),
        tool_call("fillna_with_value", bindings={"df": "all"}, func_kwargs={"columns": ["Transported"], "value": 0}),
        tool_call("drop_feature", bindings={"df": "all"}, func_kwargs={"column": "PassengerId"}),
        tool_call("encode_all_categorical_columns", bindings={"df": "all"}),
        tool_call("split_combined_into_train_test", bindings={"combined": "all"}, output="_"),
        tool_call("convert_dataframe_to_features_target", bindings={"df": "train_df"}, output="_",
                  func_kwargs={"target_column": "Transported"}),
        tool_call("convert_dataframe_to_features_target", bindings={"df": "test_df"}, output="_",
                  func_kwargs={"target_column": "Transported", "is_train": False}),
        tool_call("tune_logistic_regressor", bindings={"X_train": "X_train", "y_train": "Y_train"}, output="_",
                  func_kwargs={"cv": 2}),
        tool_call("predict_target", bindings={"model": "best_estimator", "X_data": "X_test"}, output="preds"),
    ]


def assert_valid_run(result):
    assert result["error"] is None, result["error"]
    assert result["grade"]["valid"], result["grade"]
    assert result["usage"]["calls"] > 0


@pytest.mark.parametrize("name", ["mcts_shaped", "mcts_outcome"])
def test_flat_mcts_solves_pipeline(mini_task, tmp_path, name):
    run_dir = tmp_path / "run"
    result = run_one(name, mini_task.id, run_dir, scripted_config(mcts_pipeline(mini_task)))
    assert_valid_run(result)
    stats = result["agent_stats"]
    assert stats["framework_wrote_submission"] and stats["solved_nodes"] >= 1
    assert stats["expansions"] == 12  # one candidate per step; stops at the first solved node


def test_mcts_tool_errors_become_feedback(mini_task, tmp_path):
    script = [tool_call("read_data", output="x", func_kwargs={"filepath": "/does/not/exist.csv"})]
    model = scripted_config(script)
    agent = build_agent({"name": "mcts_shaped", "max_depth": 2})
    from mltoolbench.agents.base import RunContext
    ctx = RunContext(mini_task, tmp_path / "r", model)
    stats = agent.run(ctx)
    assert stats["solved_nodes"] == 0 and not ctx.submission_path.exists()
    first = ctx.trajectory[0]["candidates"][0]
    assert first["tool"] == "read_data" and first["reward"] == 0


def test_parallel_tool_calls_branch_the_tree(mini_task, tmp_path):
    pub = mini_task.public_dir
    two = AIMessage(content="", tool_calls=[
        {"name": "read_data", "args": {"output": "a", "func_kwargs": {"filepath": str(pub / "train.csv")}},
         "id": "c1", "type": "tool_call"},
        {"name": "read_data", "args": {"output": "b", "func_kwargs": {"filepath": str(pub / "test.csv")}},
         "id": "c2", "type": "tool_call"}])
    from mltoolbench.agents.base import RunContext
    ctx = RunContext(mini_task, tmp_path / "r", scripted_config([two]))
    build_agent({"name": "mcts_shaped", "max_depth": 1}).run(ctx)
    rewards = [c["reward"] for c in ctx.trajectory[0]["candidates"]]
    assert rewards == [1.0, 0.0]  # train first is rewarded; loading test first is not


def test_lats_uses_judge_and_agents_own_submission(mini_task, tmp_path):
    run_dir = tmp_path / "run"
    from conftest import pipeline_calls  # includes save_dataframe_to_csv to the submission path
    steps = pipeline_calls(mini_task, run_dir / "submission.csv") + [AIMessage(content="The task is complete.")]

    def respond(messages, bound):
        if bound == ("Reflection",):
            final = isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls
            return tool_call("Reflection", reflections="ok", score=8 if final else 6, found_solution=final)
        return steps.pop(0)

    result = run_one("lats", mini_task.id, run_dir, scripted_config(respond))
    assert_valid_run(result)
    assert not result["agent_stats"]["framework_wrote_submission"]  # the solution node is the final text answer
    assert result["agent_stats"]["expansions"] == 13


def test_hierarchical_mcts_runs_all_stages(mini_task, tmp_path):
    # Root height at each stage's start (copies carried over) + steps that stage needs.
    limits = {"train_data_loading": 2, "test_data_loading": 3, "combine_train_test": 4, "data_cleaning": 4,
              "feature_engineering": 4, "split_train_test": 3, "train_data_to_features_target": 3,
              "test_data_to_features": 4, "modeling": 4, "create_submission_dataframe": 4}
    spec = {"name": "hierarchical_mcts", "stage_depth_limits": limits}
    result = run_one(spec, mini_task.id, tmp_path / "run", scripted_config(mcts_pipeline(mini_task)))
    assert_valid_run(result)
    stats = result["agent_stats"]
    assert stats["stage_reached"] == "create_submission_dataframe"
    assert all(n >= 1 for n in stats["solutions_per_stage"].values())


def test_hierarchical_masks_tools(mini_task, tmp_path):
    seen = []
    shared = ScriptedChatModel(script=lambda msgs, bound: (seen.append(bound), AIMessage(content="no"))[1])
    from mltoolbench.agents.base import RunContext
    from mltoolbench.llm import ModelConfig
    ctx = RunContext(mini_task, tmp_path / "r", ModelConfig(model="gpt-4.1-mini", factory=lambda _: shared))
    stats = build_agent({"name": "hierarchical_mcts", "stage_depth_limits": {"train_data_loading": 2}}).run(ctx)
    assert seen[0] == ("read_data", "get_missing_summary")
    assert stats["stage_reached"] == "train_data_loading" and not ctx.submission_path.exists()
