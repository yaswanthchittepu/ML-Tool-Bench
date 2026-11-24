"""Grade submissions and aggregate runs, following the paper's protocol.

A submission is a CSV with one row per test row, in test order. Agents write it
in whatever shape their tools produce, so the prediction column is located with
the rules below (identical to the paper's evaluation):

* ``label`` / ``value`` tasks: at most two columns. With two, prefer a numeric
  column over a boolean-like one; if both are numeric, avoid the one with all-unique
  values (an id); then prefer float; otherwise take the second column.
* ``probability`` tasks: a float column with some values strictly inside (0, 1);
  if several, ``prob_class_1.0`` or ``prob_class_1``.

A run is *valid* if its submission exists, has the right row count, a prediction
column can be located and it holds no NaN/inf. Each task reports, over N runs
(paper: 10), the *consistency* (valid / N) and the median metric and median
leaderboard percentile, counting invalid or missing runs as the worst score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mltoolbench.grading.metrics import LOWER_IS_BETTER, METRICS
from mltoolbench.tasks import Task

WORST_LOWER_IS_BETTER = 1e8  # stand-in metric for invalid runs, as in the paper


@dataclass
class Grade:
    task_id: str
    valid: bool
    metric: float | None = None
    percentile: float | None = None  # % of leaderboard entries this score beats or ties
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class InvalidSubmission(ValueError):
    pass


def grade_submission(task: Task, submission: str | Path | pd.DataFrame) -> Grade:
    answers = pd.read_csv(task.private_dir / "answers.csv")[task.target]
    try:
        df = submission if isinstance(submission, pd.DataFrame) else _read(submission)
        if len(df) != len(answers):
            raise InvalidSubmission(f"expected {len(answers)} rows, got {len(df)}")
        preds = df[find_prediction_column(df, task.prediction)]
        if preds.isna().any() or (pd.api.types.is_numeric_dtype(preds) and np.isinf(preds).any()):
            raise InvalidSubmission("predictions contain NaN or inf")
        score = METRICS[task.metric](preds, answers)
    except InvalidSubmission as e:
        return Grade(task.id, valid=False, reason=str(e))
    return Grade(task.id, valid=True, metric=score, percentile=leaderboard_percentile(task, score))


def leaderboard_percentile(task: Task, score: float) -> float | None:
    path = task.private_dir / "leaderboard.csv"
    if not path.exists():
        return None
    board = pd.read_csv(path)["Score"].dropna().to_numpy()
    beaten = board >= score if task.metric in LOWER_IS_BETTER else board <= score
    return float(beaten.mean() * 100.0)


def aggregate(task: Task, grades: list[Grade], n_runs: int = 10) -> dict:
    """Paper protocol: consistency and medians over ``n_runs``, missing runs scored as worst."""
    valid = [g for g in grades if g.valid]
    missing = max(0, n_runs - len(valid))
    worst = WORST_LOWER_IS_BETTER if task.metric in LOWER_IS_BETTER else 0.0
    metrics = [g.metric for g in valid] + [worst] * missing
    percentiles = [g.percentile for g in valid if g.percentile is not None]
    has_board = len(percentiles) == len(valid)
    return {
        "task_id": task.id,
        "metric_name": task.metric,
        "lower_is_better": task.metric in LOWER_IS_BETTER,
        "n_runs": n_runs,
        "n_valid": len(valid),
        "consistency": len(valid) / n_runs,
        "median_metric": float(np.median(metrics)),
        "median_percentile": float(np.median(percentiles + [0.0] * missing)) if has_board else None,
    }


def find_prediction_column(df: pd.DataFrame, prediction: str) -> str:
    if prediction == "probability":
        return _probability_column(df)
    cols = list(df.columns)
    if len(cols) > 2:
        raise InvalidSubmission(f"expected at most 2 columns (id + prediction), found {cols}")
    if len(cols) == 1:
        return cols[0]
    (c1, t1), (c2, t2) = ((c, _column_kind(df[c])) for c in cols)
    boolish = ("bool", "bool-like")
    if t1 == "numeric" and t2 != "numeric":
        return c1
    if t2 == "numeric" and t1 != "numeric":
        return c2
    if t1 == "numeric" and t2 == "numeric":
        n, u1, u2 = len(df), df[c1].nunique(dropna=True), df[c2].nunique(dropna=True)
        if u1 == n and u2 < n:
            return c2
        if u2 == n and u1 < n:
            return c1
        f1, f2 = pd.api.types.is_float_dtype(df[c1]), pd.api.types.is_float_dtype(df[c2])
        return c1 if f1 and not f2 else c2
    if t1 in boolish and t2 not in boolish:
        return c1
    if t2 in boolish and t1 not in boolish:
        return c2
    if t1 in boolish and t2 in boolish:
        raise InvalidSubmission(f"ambiguous: two boolean-like columns {cols}")
    raise InvalidSubmission(f"no suitable prediction column among {cols}")


def _probability_column(df: pd.DataFrame) -> str:
    candidates = []
    for col in df.columns:
        s = df[col].dropna()
        if pd.api.types.is_float_dtype(df[col]) and s.size and ((s > 1e-9) & (s < 1 - 1e-9)).any():
            candidates.append(col)
    if not candidates:
        raise InvalidSubmission("no probability column (floats strictly between 0 and 1)")
    if len(candidates) == 1:
        return candidates[0]
    for name in ("prob_class_1.0", "prob_class_1"):
        if name in candidates:
            return name
    raise InvalidSubmission(f"several probability columns {candidates}, none named prob_class_1(.0)")


def _column_kind(series: pd.Series) -> str:
    s = series.dropna()
    if s.size == 0:
        return "empty"
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s):
        return "numeric"
    if pd.api.types.is_object_dtype(s):
        values = set(s.astype(str).str.strip().str.lower().unique())
        if values and values <= {"true", "false", "yes", "no", "y", "n"}:
            return "bool-like"
    return "other"


def _read(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise InvalidSubmission(f"submission not found: {path}") from None
    except Exception as e:  # corrupted / unparsable CSV is an invalid run, not a crash
        raise InvalidSubmission(f"unreadable submission: {e}") from None
