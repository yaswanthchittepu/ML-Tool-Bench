"""Bring-your-own-agent: a user's module in their working directory, run through the runner."""
import sys
import textwrap

import pytest

from mltoolbench.runner import Experiment, run_experiment

USER_AGENT = textwrap.dedent('''
    from mltoolbench.agents import Agent
    from mltoolbench.env import ToolSession

    class MyAgent(Agent):
        name = "my_agent"

        def run(self, ctx):
            s = ToolSession()
            s.call("read_data", {"output": "test", "func_kwargs": {"filepath": str(ctx.task.public_dir / "test.csv")}})
            s.call("get_features", {"bindings": {"df": "test"}, "output": "sub",
                                    "func_kwargs": {"columns": ["PassengerId"]}})
            s.scratchpad["sub"]["Transported"] = 1
            s.call("save_dataframe_to_csv", {"bindings": {"df": "sub"},
                                             "func_kwargs": {"filepath": str(ctx.submission_path)}})
            return {"tool_calls": s.num_tool_calls}
''')


@pytest.fixture
def user_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "user_agent_mod.py").write_text(USER_AGENT)
    monkeypatch.chdir(project)  # like running `mltoolbench run --agent user_agent_mod:MyAgent` from there
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in ("", ".", str(project))])
    yield project
    sys.modules.pop("user_agent_mod", None)


def test_user_agent_from_working_directory(mini_task, user_project, tmp_path):
    summary = run_experiment(Experiment("user_agent_mod:MyAgent", [mini_task.id], None, tmp_path / "runs", n_runs=2))
    assert summary["agent_name"] == "my_agent"
    assert (tmp_path / "runs" / "my_agent" / mini_task.id / "run_1" / "result.json").exists()
    assert summary["per_task"][0]["consistency"] == 1.0


def test_bad_agent_spec_fails_fast(mini_task, tmp_path):
    with pytest.raises(ModuleNotFoundError):
        run_experiment(Experiment("no_such_module:Agent", [mini_task.id], None, tmp_path / "runs", n_runs=10))
    assert not (tmp_path / "runs").exists()  # no ten failed runs written
