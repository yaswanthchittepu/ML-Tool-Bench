"""Kaggle leaderboards shipped with the benchmark, for ranking scores as percentiles.

Each task's public leaderboard is stored in ``tasks/<task_id>/leaderboard.csv``,
reduced to a single ``Score`` column (no team names or user ids). All 15
competitions are closed, so their leaderboards are frozen.

Maintainers (re)build them with ``mltoolbench leaderboards --fetch`` (Kaggle API)
or ``mltoolbench leaderboards --from DIR`` (existing ``<task_id>.csv`` exports).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from mltoolbench.tasks import TASKS_DIR, Task

LOWER_IS_BETTER_METRICS = {"rmse", "rmsle"}


def shipped_path(task: Task) -> Path:
    return TASKS_DIR / task.id / "leaderboard.csv"


def write_leaderboard(task: Task, source: str | Path) -> int:
    """Keep only the scores of a Kaggle leaderboard export, best first. Returns the entry count."""
    board = pd.read_csv(source)
    if "Score" not in board.columns:
        raise ValueError(f"{source}: expected a 'Score' column, found {list(board.columns)}")
    scores = pd.to_numeric(board["Score"], errors="coerce").dropna()
    scores = scores.sort_values(ascending=task.metric in LOWER_IS_BETTER_METRICS)
    scores.to_frame("Score").to_csv(shipped_path(task), index=False)
    return len(scores)


def fetch_leaderboard(task: Task) -> int:
    """Download the public leaderboard with the Kaggle CLI and store its scores."""
    if shutil.which("kaggle") is None:
        raise RuntimeError("The Kaggle CLI is required: pip install 'mltoolbench[prepare]'")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["kaggle", "competitions", "leaderboard", task.kaggle, "--download", "-p", tmp], check=True)
        for archive in Path(tmp).glob("*.zip"):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp)
        csvs = sorted(Path(tmp).rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"Kaggle returned no leaderboard CSV for {task.kaggle}")
        return write_leaderboard(task, csvs[0])
