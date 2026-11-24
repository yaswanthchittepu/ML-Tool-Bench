import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.insert(0, str(Path(__file__).parent))  # make tests/fakes.py importable

from fakes import tool_call  # noqa: E402

from mltoolbench.tasks import get_task  # noqa: E402

TASK = "spaceship_titanic"  # binary / label / accuracy: the synthetic data below mimics it


@pytest.fixture
def mini_task(tmp_path, monkeypatch):
    """A tiny prepared copy of spaceship_titanic under a temporary data root."""
    monkeypatch.setenv("MLTOOLBENCH_DATA", str(tmp_path / "data"))
    get_task.cache_clear()
    task = get_task(TASK)
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "PassengerId": [f"{i:04d}_01" for i in range(n)],
        "Age": rng.normal(30, 10, n).round(),
        "HomePlanet": rng.choice(["Earth", "Mars", "Europa"], n),
        "Transported": rng.integers(0, 2, n),
    })
    df.loc[::7, "Age"] = np.nan
    task.public_dir.mkdir(parents=True)
    task.private_dir.mkdir(parents=True)
    df.iloc[:60].to_csv(task.public_dir / "train.csv", index=False)
    df.iloc[60:].drop(columns="Transported").to_csv(task.public_dir / "test.csv", index=False)
    df.iloc[60:][["PassengerId", "Transported"]].to_csv(task.private_dir / "answers.csv", index=False)
    pd.DataFrame({"Score": [0.4, 0.5, 0.6, 0.7, 0.8]}).to_csv(task.private_dir / "leaderboard.csv", index=False)
    yield task
    get_task.cache_clear()


def pipeline_calls(task, submission_path):
    """The 10-stage pipeline as tool calls, ending in a saved submission."""
    pub = task.public_dir
    return [
        tool_call("read_data", output="train", func_kwargs={"filepath": str(pub / "train.csv")}),
        tool_call("read_data", output="test", func_kwargs={"filepath": str(pub / "test.csv")}),
        tool_call("concatenate_train_test", bindings={"train_df": "train", "test_df": "test"}, output="all"),
        tool_call("fillna_with_median", bindings={"df": "all"}, func_kwargs={"columns": ["Age"]}),
        tool_call("drop_feature", bindings={"df": "all"}, output="all", func_kwargs={"column": "PassengerId"}),
        tool_call("encode_all_categorical_columns", bindings={"df": "all"}),
        tool_call("split_combined_into_train_test", bindings={"combined": "all"}, output="_"),
        tool_call("convert_dataframe_to_features_target", bindings={"df": "train_df"}, output="_",
                  func_kwargs={"target_column": "Transported"}),
        tool_call("convert_dataframe_to_features_target", bindings={"df": "test_df"}, output="_",
                  func_kwargs={"target_column": "Transported", "is_train": False}),
        tool_call("tune_logistic_regressor", bindings={"X_train": "X_train", "y_train": "Y_train"}, output="_",
                  func_kwargs={"cv": 2}),
        tool_call("predict_target", bindings={"model": "best_estimator", "X_data": "X_test"}, output="preds"),
        tool_call("save_dataframe_to_csv", bindings={"df": "preds"}, func_kwargs={"filepath": str(submission_path)}),
    ]
