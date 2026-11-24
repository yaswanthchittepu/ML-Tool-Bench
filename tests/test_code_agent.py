"""ReAct-Code: generate -> execute (subprocess) -> reflect."""
from fakes import scripted_config, tool_call

from mltoolbench.agents.code_agent import ReActCodeAgent
from mltoolbench.runner import run_one


def solution(task, submission):
    return f"""
train = pd.read_csv("{task.public_dir / 'train.csv'}")
test = pd.read_csv("{task.public_dir / 'test.csv'}")
from sklearn.linear_model import LogisticRegression
X = train[["Age"]].fillna(30)
model = LogisticRegression().fit(X, train["Transported"])
pd.DataFrame({{"PassengerId": test["PassengerId"], "Transported": model.predict(test[["Age"]].fillna(30))}}).to_csv(
    "{submission}", index=False)
print("saved")
"""


def test_fix_after_error_then_complete(mini_task, tmp_path):
    run_dir = tmp_path / "run"
    script = [
        tool_call("execute_full_solution", code="raise ValueError('bad feature')"),
        tool_call("execute_full_solution", code=solution(mini_task, run_dir / "submission.csv")),
        tool_call("mark_solution_complete"),
    ]
    result = run_one("react_code", mini_task.id, run_dir, scripted_config(script))
    assert result["error"] is None
    assert result["agent_stats"] == {"executions": 2, "successful_executions": 1, "marked_complete": True}
    assert result["grade"]["valid"], result["grade"]
    assert (run_dir / "solution_final.py").exists()


def test_iteration_cap_is_enforced(mini_task, tmp_path):
    def always_fail(*_):
        return tool_call("execute_full_solution", code="1/0")

    result = run_one({"name": "react_code", "max_iterations": 3}, mini_task.id, tmp_path / "r",
                     scripted_config(always_fail))
    assert result["agent_stats"]["executions"] == 3


def test_execution_output_format_and_timeout(tmp_path):
    agent = ReActCodeAgent(timeout_s=2)
    out = agent.execute("print(np.arange(3).sum())", tmp_path / "a.py")
    assert out.startswith("✓ CODE EXECUTED SUCCESSFULLY") and "3" in out
    out = agent.execute("x = undefined_name", tmp_path / "b.py")
    assert out.startswith("✗") and "NameError" in out and "--- TRACEBACK ---" in out
    out = agent.execute("import time; time.sleep(10)", tmp_path / "c.py")
    assert "TimeoutError" in out


def test_paper_profile_lifts_the_cap():
    assert ReActCodeAgent(profile="paper").max_iterations == 125
