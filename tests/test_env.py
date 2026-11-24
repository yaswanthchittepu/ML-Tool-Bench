"""The scratchpad environment: catalog integrity and tool-call semantics (no LLM needed)."""
import numpy as np
import pandas as pd
import pytest

from mltoolbench.env import Kind, ToolUsageError, all_tools, execute, stage_names, tools_for_stage


@pytest.fixture
def frames(tmp_path):
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"id": range(40), "a": rng.normal(size=40), "cat": rng.choice(["x", "y"], 40),
                          "y": rng.integers(0, 2, 40)})
    train.loc[3, "a"] = np.nan
    test = train.drop(columns="y").iloc[:10]
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    return tmp_path


def call(name, pad, **kw):
    return execute(all_tools()[name], pad, **kw)


def test_catalog_matches_paper():
    tools = all_tools()
    assert len(tools) == 61
    assert len(stage_names()) == 10
    for stage in stage_names():
        assert tools_for_stage(stage), stage
    assert all(spec.description.startswith("This tool") for spec in tools.values())


def test_tuple_tools_declare_outputs():
    for spec in all_tools().values():
        if spec.name.startswith("tune_"):
            assert spec.outputs == ("best_estimator", "best_params", "best_cv_score")


def test_execute_does_not_mutate_scratchpad(frames):
    pad = {}
    res = call("read_data", pad, output="train", func_kwargs={"filepath": str(frames / "train.csv")})
    assert pad == {}
    assert set(res.updates) == {"train"}
    assert res.message.startswith("Applied read_data")


def test_override_writes_back_to_its_binding(frames):
    pad = {"df": pd.read_csv(frames / "train.csv")}
    res = call("fillna_with_median", pad, bindings={"df": "df"}, func_kwargs={"columns": ["a"]})
    assert list(res.updates) == ["df"]
    assert res.updates["df"]["a"].isna().sum() == 0


def test_override_rejects_multiple_bindings():
    with pytest.raises(ToolUsageError):
        call("fillna_with_median", {}, bindings={"df": "a", "other": "b"})


def test_get_tool_returns_text_only(frames):
    pad = {"df": pd.read_csv(frames / "train.csv")}
    res = call("get_missing_summary", pad, bindings={"df": "df"})
    assert res.updates == {}
    assert "result is" in res.message


def test_full_pipeline_fixed_output_keys(frames):
    pad = {}

    def step(name, **kw):
        pad.update(call(name, pad, **kw).updates)

    step("read_data", output="train", func_kwargs={"filepath": str(frames / "train.csv")})
    step("read_data", output="test", func_kwargs={"filepath": str(frames / "test.csv")})
    step("concatenate_train_test", bindings={"train_df": "train", "test_df": "test"}, output="all")
    step("fillna_with_median", bindings={"df": "all"}, func_kwargs={"columns": ["a"]})
    step("encode_all_categorical_columns", bindings={"df": "all"})
    step("split_combined_into_train_test", bindings={"combined": "all"})
    step("convert_dataframe_to_features_target", bindings={"df": "train_df"}, func_kwargs={"target_column": "y"})
    step("convert_dataframe_to_features_target", bindings={"df": "test_df"},
         func_kwargs={"target_column": "y", "is_train": False})
    assert {"train_df", "test_df", "X_train", "Y_train", "X_test"} <= set(pad)
    step("tune_logistic_regressor", bindings={"X_train": "X_train", "y_train": "Y_train"}, func_kwargs={"cv": 3})
    assert 0.0 <= pad["best_cv_score"] <= 1.0
    step("predict_target", bindings={"model": "best_estimator", "X_data": "X_test"}, output="preds")
    assert len(pad["preds"]) == 10


def test_binding_overrides_force_key(frames):
    pad = {"X": pd.DataFrame({"a": [0.0, 1.0]}), "best_estimator": "FORCED", "other": "IGNORED"}
    spec = all_tools()["predict_target"]
    seen = {}
    fake = type(spec)(spec.name, lambda model, X_data, **_: seen.setdefault("model", model) and None, Kind.GET_SET)
    execute(fake, pad, bindings={"model": "other", "X_data": "X"}, output="p",
            binding_overrides={"model": "best_estimator"})
    assert seen["model"] == "FORCED"
