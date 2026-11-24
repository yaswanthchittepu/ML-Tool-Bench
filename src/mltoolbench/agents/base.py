"""The agent interface: implement :class:`Agent` to evaluate your own agent.

An agent gets a :class:`RunContext` for one attempt at one task. It must write its
predictions to ``ctx.submission_path`` (the task prompt tells LLM agents where),
and may record steps with ``ctx.log``. The runner handles timing, token usage,
saving the trajectory and grading.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mltoolbench.llm import ModelConfig, UsageTracker, make_chat_model
from mltoolbench.tasks import Task

SUBMISSION_FILE = "submission.csv"
_PREVIEW_CHARS = 1000


@dataclass
class RunContext:
    task: Task
    run_dir: Path
    llm: ModelConfig | None = None
    run_id: str = "run_0"
    seed: int | None = None
    verbose: bool = False
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    usage: UsageTracker | None = None

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if self.usage is None and self.llm is not None:
            self.usage = UsageTracker(self.llm.price())

    @property
    def submission_path(self) -> Path:
        return self.run_dir / SUBMISSION_FILE

    @property
    def model_dir(self) -> Path:
        return self.run_dir / "models"

    def prompt(self) -> str:
        """The task description, pointing the agent at this run's output paths."""
        return self.task.prompt(submission_dir=self.run_dir, submission_file=SUBMISSION_FILE, model_dir=self.model_dir)

    def chat_model(self, **overrides: Any):
        """A chat model for this run; all its calls are counted in ``ctx.usage``."""
        if self.llm is None:
            raise ValueError("This run has no LLM configured")
        config = self.llm if not overrides else ModelConfig(**{**self.llm.__dict__, **overrides})
        return make_chat_model(config, callbacks=[self.usage] if self.usage else None)

    def log(self, step_type: str, **data: Any) -> None:
        """Append a step to the run's trajectory (saved as trajectory.json)."""
        step = {"step": len(self.trajectory) + 1, "time": datetime.now().isoformat(), "type": step_type}
        step.update({k: _preview(v) for k, v in data.items()})
        self.trajectory.append(step)
        if self.verbose:
            print(f"[{self.task.id}/{self.run_id}] step {step['step']}: {step_type}")


class Agent(ABC):
    #: Registry name, e.g. "react". Results are stored under this name.
    name: str = "agent"

    @abstractmethod
    def run(self, ctx: RunContext) -> dict[str, Any] | None:
        """Attempt the task. Return optional extra stats to store in result.json."""


def _preview(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _PREVIEW_CHARS:
        return value[:_PREVIEW_CHARS] + f"... [{len(value)} chars]"
    try:
        json.dumps(value)
        return value
    except TypeError:
        return _preview(str(value))
