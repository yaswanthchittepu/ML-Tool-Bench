"""Verifiable stage rewards (MCTS-Shaped / MCTS-Outcome / Hierarchical MCTS) and the
LATS reflection judge's output model.

Checks read the candidate's messages (``[AIMessage with
one tool call, ToolMessage, ...]``) and the scratchpad *delta* its tool produced;
they rely on the ``"Applied <tool>"`` message prefix and the fixed output keys of
:mod:`mltoolbench.env.tools`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

STAGES = [
    "train_data_loading", "test_data_loading", "combine_train_test", "data_cleaning", "feature_engineering",
    "split_train_test", "train_data_to_features_target", "test_data_to_features", "modeling",
    "create_submission_dataframe",
]
SHAPED_REWARDS = {
    "train_data_loading": 1, "test_data_loading": 1, "combine_train_test": 1, "data_cleaning": 2,
    "feature_engineering": 2, "split_train_test": 1, "train_data_to_features_target": 1,
    "test_data_to_features": 1, "modeling": 2, "create_submission_dataframe": 2,
}
OUTCOME_STAGES = ["modeling", "create_submission_dataframe"]
OUTCOME_REWARDS = {"modeling": 0, "create_submission_dataframe": 2}
CV_SCALE = 2.5
_NUMERIC_TYPES = ["int64", "int32", "float64", "float32", "int", "float"]

ProblemType = Literal["binary-acc", "binary-probs", "regression", "multiclass"]


def problem_type(task) -> ProblemType:
    """The problem-type label the submission check is keyed on."""
    if task.task_type == "binary":
        return "binary-probs" if task.prediction == "probability" else "binary-acc"
    return task.task_type


@dataclass
class Verdict:
    stage: str | None
    reward: float
    feedback: str

    @property
    def solved(self) -> bool:
        return self.reward > 0

    def message(self) -> HumanMessage:
        return HumanMessage(content=self.feedback)


class StageVerifier:
    """Scores one candidate step against the next unsolved stage.

    ``mode``: ``shaped`` (all 10 stages), ``outcome`` (modeling + submission only),
    ``hierarchical`` (shaped rewards, evaluated for an explicitly given stage, with
    the hierarchical script's feedback wording and a -100 bonus for a NaN CV score).
    """

    def __init__(self, problem: ProblemType, mode: Literal["shaped", "outcome", "hierarchical"] = "shaped"):
        self.problem = problem
        self.mode = mode
        self.stages = OUTCOME_STAGES if mode == "outcome" else STAGES
        self.rewards = OUTCOME_REWARDS if mode == "outcome" else SHAPED_REWARDS
        self.nan_cv_bonus = {"shaped": 0.0, "outcome": 0.0, "hierarchical": -100.0}[mode]

    def next_stage(self, stages_solved: list[str]) -> str | None:
        done = [self.stages.index(s) for s in stages_solved if s in self.stages]
        index = max(done, default=-1) + 1
        return self.stages[index] if index < len(self.stages) else None

    def check(self, stage: str | None, msgs: list[BaseMessage], delta: dict[str, Any] | None, tool_doc: str) -> Verdict:
        delta = delta or {}
        if stage is None:
            prefix = "Human Feedback: " if self.mode == "hierarchical" else ""
            return Verdict(None, 0.0, f"{prefix}The trajectory has already been solved. No further rewards will be granted.")
        return getattr(self, f"_{stage}")(stage, msgs, delta, tool_doc)

    # ------------------------------------------------------------------ helpers
    def _r(self, stage: str) -> float:
        return float(self.rewards[stage])

    @staticmethod
    def _call(msgs: list[BaseMessage]) -> dict:
        return msgs[0].additional_kwargs["tool_calls"][0]

    @staticmethod
    def _tool_output(msgs: list[BaseMessage]) -> str:
        return msgs[1].content  # IndexError when the step made no tool call: handled by callers

    @staticmethod
    def _retry(what: str, doc: str, prefix: str = "Human feedback: ") -> str:
        return (f"{prefix}{what}. Please check the signature of the wrapped function and call it with the correct "
                f"arguments. The tool signature is {doc}")

    def _read(self, msgs, csv: str) -> bool:
        call = self._call(msgs)
        return (call["name"] == "read_data" and csv in call["args"]["func_kwargs"]["filepath"]
                and "Applied read_data " in self._tool_output(msgs))

    # ------------------------------------------------------------------ stages
    def _train_data_loading(self, stage, msgs, delta, doc):
        not_loaded = self._retry("Verified that the train data was NOT loaded successfully", doc)
        try:
            if self._read(msgs, "train.csv"):
                return Verdict(stage, self._r(stage), "Human feedback: Verified that the train data was loaded successfully")
            if self.mode != "hierarchical" and self._read(msgs, "test.csv"):
                return Verdict(stage, 0.0, "Human feedback: Loaded test data before train. Penalized for this")
        except Exception:
            pass
        return Verdict(stage, 0.0, not_loaded)

    def _test_data_loading(self, stage, msgs, delta, doc):
        try:
            if self._read(msgs, "test.csv"):
                return Verdict(stage, self._r(stage), "Human feedback: Verified that the test data was loaded successfully")
        except Exception:
            pass
        return Verdict(stage, 0.0, self._retry("Verified that the test data was NOT loaded successfully", doc))

    def _combine_train_test(self, stage, msgs, delta, doc):
        try:
            if "Applied concatenate_train_test" in self._tool_output(msgs):
                return Verdict(stage, self._r(stage),
                               "Human feedback: Verified that the train and test data were concatenated successfully")
        except Exception:
            pass
        return Verdict(stage, 0.0, self._retry("Verified that the train and test data were NOT concatenated successfully", doc))

    def _data_cleaning(self, stage, msgs, delta, doc):
        try:
            assert len(delta) == 1, "The Scratchpad in the data_cleaning phase is expected to only have 1 item"
            df = next(iter(delta.values()))
            if not df.isnull().any().any():
                return Verdict(stage, self._r(stage), "Human Feedback: Verified that the data cleaning was successful and "
                               "no Nans were found in non-target columns at the current step")
            null_cols = df.columns[df.isnull().any()].tolist()
            return Verdict(stage, 0.0, f"Human Feedback: Verified that the data cleaning is still in progress. "
                                       f"Nulls found in columns: {null_cols}")
        except Exception:
            return Verdict(stage, 0.0, self._retry("Data Cleaning is still in progress", doc, prefix=""))

    def _feature_engineering(self, stage, msgs, delta, doc):
        try:
            assert len(delta) == 1, "The Scratchpad in the feature_engineering phase is expected to only have 1 item"
            df = next(iter(delta.values()))
            if len(df.columns) > 100000:
                return Verdict(stage, 0.0, "Human Feedback: Verified that the feature engineering is incomplete and number "
                               "of columns is greater than 100000. Issue is probably from encoding category columns "
                               "with high cardinality. Please consider dropping these columns instead of encoding them.")
            if all(str(df[c].dtype) in _NUMERIC_TYPES or pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
                return Verdict(stage, self._r(stage), "Human Feedback: Verified that the feature engineering is complete "
                               "and all columns are numeric (int/float)")
            categorical, other = [], []
            for col in df.columns:
                dtype, unique = str(df[col].dtype), df[col].nunique()
                if isinstance(df[col].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(df[col]):
                    categorical.append({"column": col, "unique_count": unique, "dtype": dtype})
                elif not (dtype in _NUMERIC_TYPES or pd.api.types.is_numeric_dtype(df[col])):
                    other.append({"column": col, "dtype": dtype, "unique_count": unique})
            return Verdict(stage, 0.0, f"Categorical columns found: {categorical}. Columns with dtypes that are not "
                           f"numeric/categorical found: {other}. Please consider casting them to the correct dtypes or "
                           f"encoding them. Consider dropping categorical columns with a large number of categories, i.e "
                           f"exceeding 100. Only numeric columns are allowed at the end of feature engineering stage.")
        except Exception:
            return Verdict(stage, 0.0, self._retry("Feature Engineering is still in progress", doc, prefix=""))

    def _split_train_test(self, stage, msgs, delta, doc):
        try:
            if "Applied split_combined_into_train_test" in self._tool_output(msgs):
                return Verdict(stage, self._r(stage), "Human Feedback: Verified that the train and test data were split successfully")
        except Exception:
            pass
        return Verdict(stage, 0.0, self._retry("Verified that the train and test data were NOT split successfully", doc,
                                               prefix="Human Feedback: "))

    def _features(self, stage, msgs, delta, doc, *, train: bool):
        try:
            if "Applied convert_dataframe_to_features_target" not in self._tool_output(msgs):
                return Verdict(stage, 0.0, self._retry("convert_dataframe_to_features_target tool not successfully applied",
                                                       doc, prefix="Human Feedback: "))
            if train and "X_train" in delta and "Y_train" in delta and len(delta) == 2:
                return Verdict(stage, self._r(stage), "Human Feedback: Verified that the train data was successfully "
                               "converted to features (X_train) and target (Y_train)")
            if not train and "X_test" in delta and len(delta) == 1:
                return Verdict(stage, self._r(stage), "Human Feedback: Verified that the test data was successfully "
                               "converted to features (X_test)")
            what = "missing X_train and/or Y_train" if train else "missing X_test or has extra items"
            return Verdict(stage, 0.0, f"Human Feedback: Tool applied but scratchpad {what}. Found keys: {list(delta)}")
        except Exception as e:
            kind = "train data to features/target" if train else "test data to features"
            return Verdict(stage, 0.0, self._retry(f"Error checking {kind} conversion: {e}", doc, prefix="Human Feedback: "))

    def _train_data_to_features_target(self, stage, msgs, delta, doc):
        return self._features(stage, msgs, delta, doc, train=True)

    def _test_data_to_features(self, stage, msgs, delta, doc):
        return self._features(stage, msgs, delta, doc, train=False)

    def _modeling(self, stage, msgs, delta, doc):
        hier = self.mode == "hierarchical"
        pending = self._retry("Modeling is still in progress", doc, prefix="") if hier else "Modeling is still in progress"
        try:
            output = self._tool_output(msgs)
            tuned = output.startswith("Applied tune_")
            if not tuned:
                return Verdict(stage, 0.0, pending)
            bonus = 0.0
            if "best_cv_score" in delta:
                cv = float(np.asarray(delta["best_cv_score"]).item())
                bonus = self.nan_cv_bonus if math.isnan(cv) else cv * CV_SCALE
            return Verdict(stage, self._r(stage) + bonus, "Human Feedback: Verified that the modeling was successful")
        except Exception:
            what = "Modeling is not yet completed successfully" if hier else "Modeling is still in progress"
            return Verdict(stage, 0.0, self._retry(what, doc, prefix=""))

    def _create_submission_dataframe(self, stage, msgs, delta, doc):
        failed = ("Human Feedback: submission DataFrame was NOT created successfully. Please check the signature of the "
                  f"wrapped function if any, and call it with the correct arguments. The tool signature is {doc}")
        try:
            for value in delta.values():
                if not (hasattr(value, "columns") and hasattr(value, "shape")):
                    continue
                if value.isnull().any().any():
                    return Verdict(stage, 0.0, self._retry("Submission DataFrame creation failed", doc, prefix="Human Feedback: "))
                ok, kind = validate_submission_frame(value, self.problem)
                if ok:
                    return Verdict(stage, self._r(stage), f"Human Feedback: Verified that the submission DataFrame was "
                                   f"created successfully with {len(value.columns)} columns ({kind}) and no missing values.")
                expected = ("1 (single prob), 2 (prob_0 + prob_1), or 3 (ID + prob_0 + prob_1)"
                            if self.problem == "binary-probs" else "1 (prediction) or 2 (ID + prediction)")
                return Verdict(stage, 0.0, f"Human Feedback: Submission DataFrame validation failed for {self.problem}. "
                                           f"Has {len(value.columns)} columns, expected {expected}.")
        except Exception as e:
            return Verdict(stage, 0.0, self._retry(f"Unable to verify submission DataFrame creation: {e}", doc,
                                                   prefix="Human Feedback: "))
        return Verdict(stage, 0.0, failed)


# ---------------------------------------------------------------------- submission check

def validate_submission_frame(df: pd.DataFrame, problem: str) -> tuple[bool, str]:
    """Does ``df`` look like a submission for ``problem``? Returns (valid, prediction kind)."""
    cols = list(df.columns)
    if problem == "binary-probs":
        probs = sum(_valid_column(df[c], "binary-probs")[0] for c in cols)
        if len(cols) == 3 and probs >= 2:
            return True, "binary-probs-3col"
        if len(cols) == 2 and probs == 2:
            return True, "binary-probs-2col"
        if len(cols) == 1 and probs == 1:
            return True, "binary-probs-1col"
        return False, "unknown"
    kind = "unknown"
    if len(cols) in (1, 2):
        for c in cols:
            ok, kind = _valid_column(df[c], problem)
            if ok:
                return True, kind
    return False, kind


def _valid_column(col: pd.Series, problem: str | None) -> tuple[bool, str]:
    values = col.unique()
    if problem == "binary-acc":
        if all(v in [True, False] for v in values):
            return True, "boolean"
        if all(v in [0, 1] for v in values):
            return True, "binary"
        return False, "invalid"
    if problem == "binary-probs":
        if all(isinstance(v, int | float) and 0 <= v <= 1 for v in values):
            return True, "probability"
        return False, "invalid"
    if problem == "regression":
        if all(isinstance(v, int | float | np.number) for v in values):
            return True, "regression"
        return False, "invalid"
    if problem == "multiclass":
        if all(isinstance(v, int | np.integer) for v in values) and len(values) <= 50:
            return True, "multiclass"
        return False, "invalid"
    if all(v in [True, False] for v in values):
        return True, "boolean"
    if all(v in [0, 1] for v in values):
        return True, "binary"
    if all(isinstance(v, int | float) and 0 <= v <= 1 for v in values):
        return True, "probability"
    if all(isinstance(v, int | np.integer) for v in values) and len(values) <= 20:
        return True, "multiclass"
    if all(isinstance(v, int | float | np.number) for v in values):
        return True, "regression"
    return False, "invalid"


# ---------------------------------------------------------------------- LATS judge

class Reflection(BaseModel):
    reflections: str = Field(
        description="The critique and reflections on the sufficiency, superfluency,"
        " and general quality of the response"
    )
    score: int = Field(description="Score from 0-10 on the quality of the candidate response.", gte=0, lte=10)
    found_solution: bool = Field(description="Whether the response has fully solved the question or task.")

    def as_message(self) -> HumanMessage:
        return HumanMessage(content=f"Reasoning: {self.reflections}\nScore: {self.score}")

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0
