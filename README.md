# ML-Tool-Bench

**A benchmark for tool-augmented planning on tabular ML tasks.**
Paper: [ML-Tool-Bench: Tool-Augmented Planning for ML Tasks](https://arxiv.org/abs/2512.00672)

Agents solve 15 Kaggle challenges end to end (load, clean, engineer features, model and submit) using
**61 ML tools** instead of writing code. Data never passes through the LLM. Every dataframe and
model lives in a *scratchpad* (a named-object store), and tools read and write it by name. Solving a
task takes dozens of correctly chained tool calls, so this tests long-horizon planning more than ML
knowledge.

| | |
|---|---|
| **Tasks** | 15 Kaggle competitions: 7 regression, 6 binary, 2 multiclass. Each uses a 10k-row sample with 20% held out. |
| **Action space** | 61 tools in 6 groups: io, cleaning, features, dataframe, tuning, evaluation |
| **Score** | Kaggle leaderboard percentile (median over 10 runs) and consistency (fraction of runs with a valid submission) |
| **Baselines** | ReAct, LATS, MCTS-Outcome, MCTS-Shaped, Hierarchical MCTS, ReAct-Code |

## Install

```bash
git clone https://github.com/yaswanthchittepu/ML-Tool-Bench.git && cd ML-Tool-Bench
pip install -e ".[agents,prepare]"      # Python >= 3.11
cp .env.example .env                    # set OPENAI_MODEL and OPENAI_API_KEY
```

- macOS: xgboost and LightGBM need OpenMP. Install it with `brew install libomp`.
- The ML stack (pandas, scikit-learn, xgboost, LightGBM, CatBoost) is pinned to exact versions.
  CV scores feed the MCTS rewards, so results are only comparable on these versions.
- `pip install -e .` alone installs only the tools, environment and grader. That's enough to evaluate
  your own agent.

## Prepare the data

The datasets are not redistributed. You need a [Kaggle API token](https://www.kaggle.com/docs/api) in
`~/.kaggle/kaggle.json`, and you must accept each competition's rules on kaggle.com. Then run:

```bash
mltoolbench prepare                        # all 15 tasks; or --tasks spaceship_titanic ...
```

This downloads each competition and makes the benchmark's split deterministically: up to 10,000 rows
sampled with seed 42, classification targets label-encoded, and 20% held out (stratified for
classification). Files go under `$MLTOOLBENCH_DATA` (default `./data`):

```
data/public/<task>/{train,test}.csv    # the only directory agents are told about
data/private/<task>/answers.csv        # used by the grader only
```

**Leaderboards ship with the benchmark.** Each task's public Kaggle leaderboard is stored as
`src/mltoolbench/tasks/<task>/leaderboard.csv`. It contains scores only, with no team names, so
percentiles need no Kaggle access. The files are a snapshot taken on 2026-11-24. Fourteen of the
competitions are closed, so their leaderboards no longer change. Spaceship Titanic is an ongoing
"Getting Started" competition, so its live leaderboard keeps moving; the benchmark always uses the
snapshot.
To use a different leaderboard, put it at `data/private/<task>/leaderboard.csv`. Maintainers rebuild
the shipped files with `mltoolbench leaderboards --fetch` (Kaggle API) or `--from DIR`
(existing `<task>.csv` exports).

## Run a baseline

```bash
mltoolbench run --config baselines/mcts_shaped.yaml --model gpt-4o --workers 8
mltoolbench run --agent react --tasks spaceship_titanic --runs 3      # quick try
mltoolbench report runs/                                              # results table
```

Each run's output lands in `runs/<model>/<agent>/<task>/run_<i>/`:

- `submission.csv`
- `trajectory.json`: every LLM step and tool call
- `result.json`: grade, tokens, cost and wall time

Each agent also gets a `summary.json` with per-task aggregates.

| Config | Agent |
|---|---|
| `baselines/react.yaml` | ReAct: plan, then one tool call per turn |
| `baselines/lats.yaml` | LATS: MCTS with an LLM-reflection value |
| `baselines/mcts_outcome.yaml` | MCTS that rewards only a trained model and a valid submission |
| `baselines/mcts_shaped.yaml` | MCTS with shaped, verifiable stage rewards and feedback |
| `baselines/hierarchical_mcts.yaml` | Stage-wise MCTS with tool masking |
| `baselines/react_code.yaml` | Writes and executes Python instead of using tools (contrast baseline) |

To reproduce the exact numbers reported in the paper, use the `paper-v1` release (`git checkout paper-v1`).

Any OpenAI-compatible endpoint works, including vLLM, Azure and LiteLLM. Set `--base-url` or
`OPENAI_BASE_URL`. Open models that emit tool calls as JSON text are handled.

## Evaluate your own agent

**Option 1: plug into the runner.** Subclass `Agent`. You get the prompt, paths and a usage-tracked
model through `RunContext`:

```python
from mltoolbench.agents import Agent, RunContext
from mltoolbench.env import ToolSession

class MyAgent(Agent):
    name = "my_agent"

    def run(self, ctx: RunContext):
        session = ToolSession()               # the 61 tools + a fresh scratchpad
        tools = session.openai_tools()        # OpenAI function-calling schemas
        prompt = ctx.prompt()                 # task description, says where to save the submission
        ...                                   # your loop: result_text = session.call(name, args)
        # the submission must end up at ctx.submission_path
```

```bash
mltoolbench run --agent my_package.my_module:MyAgent --model gpt-4o
```

**Option 2: use only the pieces.** `ToolSession` works with any framework, or with raw API calls. To
score a submission, run `mltoolbench grade <task> submission.csv`, or call
`mltoolbench.grading.grade_submission`. LangGraph users can call
`mltoolbench.env.langchain.langchain_tools(...)` to get ready-made tools.

Hierarchical agents can restrict tools to a pipeline stage with `ToolSession(stage="modeling")`. The 10
stages and their tool masks are defined in `src/mltoolbench/tools/catalog.yaml`.

## Evaluation protocol

- A submission is **valid** when all of the following hold:
  - it has one row per test row, in test order;
  - its prediction column can be located;
  - that column contains no NaN or inf values.

  How the prediction column is located is documented in `grading/grader.py`.
- The metric is each competition's official metric: RMSE, RMSLE, accuracy, AUC, MCC, macro-F1 or
  normalized Gini. It's converted to the **percentile** of the public Kaggle leaderboard that the
  score beats.
- For RMSLE tasks, negative predictions are clipped to 0 before scoring. The run stays valid, the
  error still counts, and `result.json` records how many values were clipped.
- Per task over N = 10 runs, **consistency** is the fraction of valid runs. The **median percentile**
  counts invalid runs as 0.
- The headline number is the median across tasks.

## Results (GPT-4o, 15 tasks, 10 runs each)

| Method | Consistency | Median percentile |
|---|---|---|
| ReAct | 0.6 | 0.58 |
| LATS | 0.6 | 7.17 |
| MCTS-Outcome | 0.6 | 7.12 |
| MCTS-Shaped | 0.8 | 9.36 |
| Hierarchical MCTS | 0.7 | **17.10** |

Per-task results and GPT-4.1-mini numbers are in the [paper](https://arxiv.org/abs/2512.00672).

## Repository layout

```
src/mltoolbench/
  tools/        the 61 tools (plain pandas/sklearn) + catalog.yaml (scratchpad kinds, stage masks)
  env/          scratchpad execution, ToolSession, LangGraph adapter
  tasks/        one folder per task (task.yaml + prompt), registry, data preparation
  grading/      metrics, grader, aggregation
  agents/       Agent interface, ReAct, ReAct-Code, MCTS family (agents/mcts/)
  runner.py     parallel experiment runner
  cli.py        the `mltoolbench` command
baselines/      experiment configs, one per baseline
tests/          API-free tests; agents run end to end against a scripted LLM
```

Run the tests with `pip install -e ".[agents,dev]" && pytest`.

## License

Code is released under the [MIT License](LICENSE). The Kaggle datasets are not redistributed; they are
downloaded under each competition's own terms.

## Citation

```bibtex
@article{chittepu2025mltoolbench,
  title   = {ML-Tool-Bench: Tool-Augmented Planning for ML Tasks},
  author  = {Chittepu, Yaswanth and Addanki, Raghavendra and Mai, Tung and Rao, Anup and Kveton, Branislav},
  journal = {arXiv preprint arXiv:2512.00672},
  year    = {2025}
}
```
