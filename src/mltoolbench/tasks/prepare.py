"""Build the benchmark's data from the original Kaggle competitions.

The split reproduces the paper exactly: sample at most 10,000 rows of Kaggle's
``train.csv`` (seed 42), label-encode classification targets, then hold out 20%
(stratified for classification, seed 42) as the benchmark's test set.

Downloading needs the Kaggle CLI (``pip install mltoolbench[prepare]``), an API
token in ``~/.kaggle/kaggle.json``, and the competition rules accepted on kaggle.com.
"""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from mltoolbench.tasks import Task

MAX_ROWS = 10_000
TEST_FRACTION = 0.2
SEED = 42


def prepare_task(task: Task, *, download: bool = True, force: bool = False) -> None:
    """Download (optionally), split and write public/private files for one task."""
    if (task.public_dir / "train.csv").exists() and not force:
        print(f"[{task.id}] already prepared (use force=True to rebuild)")
        return
    if download:
        download_competition(task)
        download_leaderboard(task)
    raw_train = _find_csv(task.raw_dir, "train.csv")
    train, test, answers = split(pd.read_csv(raw_train), task)
    task.public_dir.mkdir(parents=True, exist_ok=True)
    task.private_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(task.public_dir / "train.csv", index=False)
    test.to_csv(task.public_dir / "test.csv", index=False)
    answers.to_csv(task.private_dir / "answers.csv", index=False)
    print(f"[{task.id}] train {train.shape}, test {test.shape} -> {task.public_dir}")


def split(df: pd.DataFrame, task: Task) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, test, answers). ``answers`` holds the id and target of each test row, in order."""
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=SEED)
    X, y = df.drop(columns=task.target), df[task.target]
    if task.is_classification:
        y = pd.Series(LabelEncoder().fit_transform(y), index=y.index, name=task.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=SEED, stratify=y if task.is_classification else None,
    )
    train = pd.concat([X_train, y_train], axis=1)
    answers = pd.concat([X_test[[task.id_column]], y_test], axis=1)
    return train, X_test, answers


def import_legacy_layout(task: Task, legacy_root: str | Path, leaderboards: str | Path | None = None) -> None:
    """Adopt data produced by the paper's original scripts (``<root>/<task>/train.csv``,
    ``test.csv``, ``test_with_labels.csv``) instead of re-downloading from Kaggle."""
    src = Path(legacy_root) / task.id
    task.public_dir.mkdir(parents=True, exist_ok=True)
    task.private_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "test.csv"):
        shutil.copy(src / name, task.public_dir / name)
    labels = pd.read_csv(src / "test_with_labels.csv")
    labels[[task.id_column, task.target]].to_csv(task.private_dir / "answers.csv", index=False)
    if leaderboards and (board := Path(leaderboards) / f"{task.id}.csv").exists():
        shutil.copy(board, task.private_dir / "leaderboard.csv")
    print(f"[{task.id}] imported from {src}")


def download_competition(task: Task) -> None:
    task.raw_dir.mkdir(parents=True, exist_ok=True)
    _kaggle("competitions", "download", "-c", task.kaggle, "-p", str(task.raw_dir))
    _extract_all(task.raw_dir)


def download_leaderboard(task: Task) -> None:
    """Fetch the public leaderboard used to convert scores into percentiles."""
    tmp = task.raw_dir / "leaderboard"
    tmp.mkdir(parents=True, exist_ok=True)
    _kaggle("competitions", "leaderboard", task.kaggle, "--download", "-p", str(tmp))
    _extract_all(tmp)
    task.private_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(_find_csv(tmp, "*leaderboard*.csv"), task.private_dir / "leaderboard.csv")


def _kaggle(*args: str) -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError("The Kaggle CLI is required: pip install 'mltoolbench[prepare]'")
    subprocess.run(["kaggle", *args], check=True)


def _extract_all(directory: Path) -> None:
    """Unzip archives in place, including archives nested inside archives."""
    while archives := [p for p in directory.rglob("*.zip") if not p.with_suffix(".extracted").exists()]:
        for archive in archives:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(archive.parent)
            archive.with_suffix(".extracted").touch()


def _find_csv(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} under {directory}")
    return matches[0]
