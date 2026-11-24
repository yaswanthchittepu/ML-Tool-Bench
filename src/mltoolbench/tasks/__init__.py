"""Task registry: the 15 Kaggle challenges of ML-Tool-Bench.

Each task lives in ``tasks/<task_id>/`` as ``task.yaml`` (metadata) and
``description.md`` (the prompt given to agents, verbatim from the paper).

Data layout under the data root (``$MLTOOLBENCH_DATA`` or ``./data``)::

    raw/<task_id>/       Kaggle download (prepare step only)
    public/<task_id>/    train.csv, test.csv   <- the only directory agents see
    private/<task_id>/   answers.csv, leaderboard.csv   <- grader only
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import yaml

TASKS_DIR = Path(__file__).resolve().parent


def data_root() -> Path:
    return Path(os.environ.get("MLTOOLBENCH_DATA", "data"))


@dataclass(frozen=True)
class Task:
    id: str
    name: str
    kaggle: str
    task_type: Literal["binary", "multiclass", "regression"]
    #: What the submission's prediction column holds: class labels, P(class 1), or a value.
    prediction: Literal["label", "probability", "value"]
    metric: str
    target: str
    id_column: str
    description_template: str

    @property
    def is_classification(self) -> bool:
        return self.task_type != "regression"

    @property
    def public_dir(self) -> Path:
        return data_root() / "public" / self.id

    @property
    def private_dir(self) -> Path:
        return data_root() / "private" / self.id

    @property
    def raw_dir(self) -> Path:
        return data_root() / "raw" / self.id

    def prompt(self, submission_dir: str | Path, submission_file: str, model_dir: str | Path) -> str:
        """The task description an agent receives, with paths filled in."""
        return self.description_template.format(
            data_dir=self.public_dir, model_dir=model_dir,
            save_directory=submission_dir, save_file_name=submission_file,
        )


@cache
def get_task(task_id: str) -> Task:
    path = TASKS_DIR / task_id
    if not (path / "task.yaml").is_file():
        raise KeyError(f"Unknown task {task_id!r}. Available: {', '.join(list_tasks())}")
    meta = yaml.safe_load((path / "task.yaml").read_text())
    return Task(id=task_id, description_template=(path / "description.md").read_text(), **meta)


def list_tasks() -> list[str]:
    return sorted(p.parent.name for p in TASKS_DIR.glob("*/task.yaml"))


__all__ = ["Task", "data_root", "get_task", "list_tasks"]
