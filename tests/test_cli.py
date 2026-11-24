"""CLI commands that need no LLM."""
import pandas as pd
from fakes import scripted_config
from langchain_core.messages import AIMessage

from mltoolbench.cli import _parse_sets, main
from mltoolbench.runner import Experiment, run_experiment


def test_list(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "spaceship_titanic" in out and "hierarchical_mcts" in out


def test_grade(mini_task, tmp_path, capsys):
    sub = tmp_path / "s.csv"
    pd.DataFrame({"Transported": [1] * 20}).to_csv(sub, index=False)
    assert main(["grade", mini_task.id, str(sub)]) == 0
    assert '"valid": true' in capsys.readouterr().out


def test_prepare_from_legacy_layout(tmp_path, monkeypatch):
    legacy = tmp_path / "old" / "spaceship_titanic"
    legacy.mkdir(parents=True)
    pd.DataFrame({"PassengerId": ["a"], "Transported": [1]}).to_csv(legacy / "train.csv", index=False)
    pd.DataFrame({"PassengerId": ["b"]}).to_csv(legacy / "test.csv", index=False)
    pd.DataFrame({"PassengerId": ["b"], "x": [0], "Transported": [0]}).to_csv(legacy / "test_with_labels.csv",
                                                                             index=False)
    monkeypatch.setenv("MLTOOLBENCH_DATA", str(tmp_path / "data"))
    assert main(["prepare", "--tasks", "spaceship_titanic", "--from-legacy", str(tmp_path / "old")]) == 0
    answers = pd.read_csv(tmp_path / "data/private/spaceship_titanic/answers.csv")
    assert list(answers.columns) == ["PassengerId", "Transported"]


def test_report_table(mini_task, tmp_path, capsys):
    run_experiment(Experiment("react", [mini_task.id], scripted_config(lambda *_: AIMessage("no")), tmp_path,
                              n_runs=2))
    assert main(["report", str(tmp_path), "--csv", str(tmp_path / "t.csv")]) == 0
    out = capsys.readouterr().out
    assert "OVERALL" in out and "0.00 (0.0)" in out
    assert (tmp_path / "t.csv").exists()


def test_parse_sets():
    assert _parse_sets(["profile=paper", "max_depth=50", "use_feedback=false"]) == {
        "profile": "paper", "max_depth": 50, "use_feedback": False}
