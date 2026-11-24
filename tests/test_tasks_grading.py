"""Task registry, data split and grading protocol."""
import numpy as np
import pandas as pd
import pytest

from mltoolbench.grading import aggregate, find_prediction_column, grade_submission
from mltoolbench.grading.grader import InvalidSubmission
from mltoolbench.grading.metrics import METRICS
from mltoolbench.tasks import get_task, list_tasks
from mltoolbench.tasks.prepare import split


def test_registry_has_the_papers_15_tasks():
    tasks = list_tasks()
    assert len(tasks) == 15
    for t in map(get_task, tasks):
        assert t.metric in METRICS
        for placeholder in ("{data_dir}", "{model_dir}", "{save_directory}", "{save_file_name}"):
            assert placeholder in t.description_template, (t.id, placeholder)


def test_prompt_renders_paths():
    t = get_task("spaceship_titanic")
    text = t.prompt("subs/x", "run.csv", "models/x")
    assert "subs/x" in text and "run.csv" in text and "{" not in text.replace("{{", "")
    assert str(t.public_dir) in text


def test_unknown_task():
    with pytest.raises(KeyError):
        get_task("nope")


def test_split_is_deterministic_and_encodes_labels():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"Id": range(500), "x": rng.normal(size=500), "Cover_Type": rng.integers(1, 8, 500)})
    t = get_task("forest_cover_type_prediction")
    a, b = split(df, t), split(df, t)
    for x, y in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(x, y)
    train, test, answers = a
    assert len(test) == len(answers) == 100
    assert sorted(train["Cover_Type"].unique()) == list(range(7))  # 1..7 -> 0..6, as in the paper
    assert "Cover_Type" not in test
    assert list(answers.columns) == ["Id", "Cover_Type"]


@pytest.fixture
def graded_task(tmp_path, monkeypatch):
    monkeypatch.setenv("MLTOOLBENCH_DATA", str(tmp_path))
    get_task.cache_clear()
    t = get_task("spaceship_titanic")
    t.private_dir.mkdir(parents=True)
    pd.DataFrame({"PassengerId": ["a", "b", "c", "d"], "Transported": [1, 0, 1, 1]}).to_csv(
        t.private_dir / "answers.csv", index=False)
    pd.DataFrame({"Score": [0.5, 0.7, 0.8, 0.9]}).to_csv(t.private_dir / "leaderboard.csv", index=False)
    yield t
    get_task.cache_clear()


def test_grade_valid_submission(graded_task):
    sub = pd.DataFrame({"PassengerId": ["a", "b", "c", "d"], "Transported": [True, False, True, False]})
    g = grade_submission(graded_task, sub)
    assert g.valid and g.metric == 0.75 and g.percentile == 50.0


@pytest.mark.parametrize("sub,reason", [
    (pd.DataFrame({"Transported": [1, 0]}), "rows"),
    (pd.DataFrame({"a": [1] * 4, "b": [1] * 4, "c": [1] * 4}), "at most 2"),
    (pd.DataFrame({"Transported": [1.0, np.nan, 1.0, 1.0]}), "NaN"),
])
def test_grade_invalid_submissions(graded_task, sub, reason):
    g = grade_submission(graded_task, sub)
    assert not g.valid and reason in g.reason


def test_missing_file_is_invalid_not_a_crash(graded_task, tmp_path):
    assert not grade_submission(graded_task, tmp_path / "missing.csv").valid


def test_aggregate_counts_missing_runs_as_worst(graded_task):
    sub = pd.DataFrame({"Transported": [1, 0, 1, 1]})
    grades = [grade_submission(graded_task, sub)] * 4
    agg = aggregate(graded_task, grades, n_runs=10)
    assert agg["consistency"] == 0.4
    assert agg["median_metric"] == 0.0 and agg["median_percentile"] == 0.0


@pytest.mark.parametrize("df,kind,expected", [
    (pd.DataFrame({"id": [1, 2, 3], "p": [0, 1, 1]}), "label", "p"),             # id has all-unique values
    (pd.DataFrame({"id": ["a", "b"], "y": ["True", "False"]}), "label", "y"),     # bool-like strings
    (pd.DataFrame({"id": [1, 2], "prob_class_0": [0.3, 0.6], "prob_class_1": [0.7, 0.4]}), "probability", "prob_class_1"),
])
def test_find_prediction_column(df, kind, expected):
    assert find_prediction_column(df, kind) == expected


def test_probability_task_rejects_hard_labels():
    with pytest.raises(InvalidSubmission):
        find_prediction_column(pd.DataFrame({"id": [1, 2], "y": [0.0, 1.0]}), "probability")
