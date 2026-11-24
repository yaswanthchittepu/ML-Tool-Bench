"""Shipped leaderboards: scores-only export, lookup order, CLI import."""
import shutil

import pandas as pd
import pytest

import mltoolbench.tasks as tasks_mod
from mltoolbench.cli import main
from mltoolbench.grading import leaderboard_percentile
from mltoolbench.tasks import get_task
from mltoolbench.tasks.leaderboards import write_leaderboard

KAGGLE_EXPORT = pd.DataFrame({
    "Rank": [1, 2, 3, 4], "TeamId": [11, 12, 13, 14], "TeamName": ["a", "b", "c", "d"],
    "Score": [0.81, 0.79, None, 0.70], "TeamMemberUserNames": ["u1", "u2", "u3", "u4"],
})


@pytest.fixture
def package_dir(tmp_path, monkeypatch):
    """Redirect the shipped-leaderboard location to a temp copy of the task folders."""
    root = tmp_path / "pkg_tasks"
    for task_id in ("spaceship_titanic", "nyc_taxi_fare_prediction"):
        shutil.copytree(tasks_mod.TASKS_DIR / task_id, root / task_id)
    monkeypatch.setattr(tasks_mod, "TASKS_DIR", root)
    monkeypatch.setattr("mltoolbench.tasks.leaderboards.TASKS_DIR", root)
    monkeypatch.setenv("MLTOOLBENCH_DATA", str(tmp_path / "data"))
    get_task.cache_clear()
    yield root
    get_task.cache_clear()


def test_export_keeps_only_scores_best_first(package_dir, tmp_path):
    src = tmp_path / "export.csv"
    KAGGLE_EXPORT.to_csv(src, index=False)
    task = get_task("spaceship_titanic")
    assert write_leaderboard(task, src) == 3
    shipped = pd.read_csv(package_dir / "spaceship_titanic" / "leaderboard.csv")
    assert list(shipped.columns) == ["Score"]  # no team names or user ids
    assert shipped["Score"].tolist() == [0.81, 0.79, 0.70]


def test_lower_is_better_sorted_ascending(package_dir, tmp_path):
    src = tmp_path / "export.csv"
    pd.DataFrame({"Score": [3.2, 2.9, 4.0]}).to_csv(src, index=False)
    task = get_task("nyc_taxi_fare_prediction")  # rmse: lower is better
    write_leaderboard(task, src)
    assert pd.read_csv(package_dir / "nyc_taxi_fare_prediction" / "leaderboard.csv")["Score"].tolist() == [2.9, 3.2, 4.0]


def test_local_override_beats_shipped(package_dir, tmp_path):
    task = get_task("spaceship_titanic")
    pd.DataFrame({"Score": [0.5, 0.6]}).to_csv(package_dir / "spaceship_titanic" / "leaderboard.csv", index=False)
    assert task.leaderboard_path == package_dir / "spaceship_titanic" / "leaderboard.csv"
    assert leaderboard_percentile(task, 0.55) == 50.0
    task.private_dir.mkdir(parents=True)
    pd.DataFrame({"Score": [0.9, 0.95]}).to_csv(task.private_dir / "leaderboard.csv", index=False)
    assert task.leaderboard_path == task.private_dir / "leaderboard.csv"
    assert leaderboard_percentile(task, 0.55) == 0.0


def test_cli_import_from_directory(package_dir, tmp_path):
    exports = tmp_path / "exports"
    exports.mkdir()
    KAGGLE_EXPORT.to_csv(exports / "spaceship_titanic.csv", index=False)
    assert main(["leaderboards", "--tasks", "spaceship_titanic", "--from", str(exports)]) == 0
    assert (package_dir / "spaceship_titanic" / "leaderboard.csv").exists()
    assert main(["leaderboards", "--tasks", "nyc_taxi_fare_prediction", "--from", str(exports)]) == 1  # missing

