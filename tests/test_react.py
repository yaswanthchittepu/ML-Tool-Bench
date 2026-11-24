"""ReAct baselines and the runner, end to end with a scripted LLM."""
import json

from conftest import pipeline_calls
from fakes import scripted_config, tool_call
from langchain_core.messages import AIMessage

from mltoolbench.runner import Experiment, run_experiment, run_one


def test_react_solves_task_end_to_end(mini_task, tmp_path):
    run_dir = tmp_path / "run"
    script = [AIMessage(content="1. load 2. clean 3. model")]  # the plan
    script += pipeline_calls(mini_task, run_dir / "submission.csv")
    result = run_one("react", mini_task.id, run_dir, scripted_config(script))

    assert result["error"] is None
    assert result["grade"]["valid"], result["grade"]
    assert 0 <= result["grade"]["metric"] <= 1
    assert result["agent_stats"]["num_tool_calls"] == 12
    assert result["usage"]["calls"] == 14  # plan + 12 tool turns + final answer
    steps = json.loads((run_dir / "trajectory.json").read_text())["steps"]
    assert [s["type"] for s in steps].count("tool") == 12
    assert all(s["status"] == "success" for s in steps if s["type"] == "tool")


def test_react_without_plan_for_llama(mini_task, tmp_path):
    cfg = scripted_config([AIMessage(content="nothing to do")], model="meta-llama/Llama-3.1-70B")
    result = run_one("react", mini_task.id, tmp_path / "r", cfg)
    assert result["usage"]["calls"] == 1
    assert not result["grade"]["valid"]


def test_tool_errors_are_returned_to_the_agent(mini_task, tmp_path):
    seen = []

    def script(messages, _tools):
        seen.append(messages[-1])
        if len(seen) == 2:
            return tool_call("read_data", output="x", func_kwargs={"filepath": "/does/not/exist.csv"})
        return AIMessage(content="plan" if len(seen) == 1 else "giving up")

    result = run_one("react", mini_task.id, tmp_path / "r", scripted_config(script))
    assert result["error"] is None
    assert "Error" in seen[-1].content  # the failing tool call came back as a ToolMessage


def test_json_text_tool_calls_from_open_models(mini_task, tmp_path):
    path = mini_task.public_dir / "train.csv"
    text_call = AIMessage(content=json.dumps({"name": "read_data", "parameters": {
        "output": "train", "func_kwargs": str({"filepath": str(path)})}}))
    result = run_one("react", mini_task.id, tmp_path / "r",
                     scripted_config([text_call], model="llama-3.1-70b"))
    assert result["agent_stats"]["num_tool_calls"] == 1
    assert result["agent_stats"]["scratchpad_keys"] == ["train"]


def test_agent_crash_is_a_failed_run(mini_task, tmp_path):
    def boom(*_):
        raise RuntimeError("LLM endpoint down")

    result = run_one("react", mini_task.id, tmp_path / "r", scripted_config(boom))
    assert "LLM endpoint down" in result["error"]
    assert not result["grade"]["valid"]
    assert (tmp_path / "r" / "result.json").exists()


def test_hierarchical_react_walks_all_stages(mini_task, tmp_path):
    run_dir = tmp_path / "run"
    calls = pipeline_calls(mini_task, run_dir / "submission.csv")
    per_stage = [calls[0:1], calls[1:2], calls[2:3], calls[3:4], calls[4:6], calls[6:7], calls[7:8], calls[8:9],
                 calls[9:10], calls[10:12]]
    script = []
    for stage_calls in per_stage:  # plan, tool calls, then "done" ends the stage
        script += [AIMessage(content="plan"), *stage_calls, AIMessage(content="stage done")]
    result = run_one("hierarchical_react", mini_task.id, run_dir, scripted_config(script))

    assert result["error"] is None and result["agent_stats"]["stopped"] is None
    assert result["grade"]["valid"], result["grade"]
    steps = json.loads((run_dir / "trajectory.json").read_text())["steps"]
    assert sum(s["type"] == "stage_transition" for s in steps) == 9
    assert all(s["status"] == "success" for s in steps if s["type"] == "tool"), \
        [s["content"] for s in steps if s["type"] == "tool" and s["status"] != "success"]


def test_experiment_summary(mini_task, tmp_path):
    def script(messages, _tools):
        return AIMessage(content="no tools")

    summary = run_experiment(Experiment("react", [mini_task.id], scripted_config(script), tmp_path, n_runs=3))
    row = summary["per_task"][0]
    assert row["consistency"] == 0.0 and row["median_percentile"] == 0.0
    assert "api_key" not in summary["config"]["model"]
    assert (tmp_path / "react" / "summary.json").exists()
