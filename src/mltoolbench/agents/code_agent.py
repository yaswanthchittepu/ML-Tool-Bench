"""ReAct-Code: the paper's non-tool contrast baseline.

Instead of the 61 scratchpad tools, the model writes a complete Python solution,
executes it, reads the output, and iterates until it calls ``mark_solution_complete``.

Each solution runs in a fresh Python subprocess with a timeout, so a crash, hang or
memory blowup in generated code fails that attempt instead of the whole benchmark run.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from mltoolbench.agents.base import Agent, RunContext
from mltoolbench.llm import bind_tools

SUCCESS = "✓ CODE EXECUTED SUCCESSFULLY"
FAILURE = "✗ CODE EXECUTION FAILED"
PREAMBLE = "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\n"

FIRST_PROMPT = """
    You are a Data Scientist solving a Kaggle competition by writing Python code.
    Your task: Write a COMPLETE solution as a single Python script that:
    1. Loads the training and test data
    2. Performs data exploration and preprocessing
    3. Engineers features as needed
    4. Trains a machine learning model
    5. Makes predictions on test data
    6. Saves predictions to the specified CSV file
    Available packages (already installed):
    - pandas (imported as pd), numpy (imported as np), matplotlib.pyplot (imported as plt)
    - scikit-learn - all modules (ensemble, linear_model, preprocessing, model_selection, metrics, etc.)
    - xgboost, lightgbm, catboost - gradient boosting
    - matplotlib (NON-INTERACTIVE: use plt.savefig(), NEVER plt.show()), plotly
    - scipy, joblib
    Guidelines:
    - Write the ENTIRE solution in one code block
    - Add print statements to show progress and key metrics
    - Handle missing values appropriately
    - Use cross-validation to evaluate your model
    - For plots: use plt.savefig('plot.png') then plt.close(), NEVER plt.show()
    CRITICAL: You MUST call the execute_full_solution tool with your code. Do NOT just return code in markdown.
    Call: execute_full_solution(code="your entire solution code here")
    """
FIRST_ASK = "Write and execute the complete solution using the execute_full_solution tool."
REFLECT_PROMPT = """
    You are reflecting on the execution of your previous solution.
    Analyze the output/errors and:
    - If successful and predictions saved: call mark_solution_complete()
    - If errors occurred: generate IMPROVED solution using execute_full_solution(code="...")
    Available packages (already installed):
    - pandas (pd), numpy (np), matplotlib.pyplot (plt) - pre-imported
    - scikit-learn, xgboost, lightgbm, catboost
    - matplotlib (use plt.savefig(), NEVER plt.show()), plotly
    CRITICAL: Either call mark_solution_complete() OR execute_full_solution(code="...").
    Do NOT just return code in markdown - you must invoke a tool!
    """


@tool
def execute_full_solution(code: str) -> str:
    """Execute a COMPLETE Python solution and return all output."""
    raise NotImplementedError  # schema only: the agent executes calls itself


@tool
def mark_solution_complete() -> str:
    """Call when satisfied with your solution."""
    return "Solution marked as complete. Execution will terminate."


class ReActCodeAgent(Agent):
    name = "react_code"

    def __init__(self, max_iterations: int | None = 10, timeout_s: float = 3600):
        self.max_iterations = max_iterations
        self.timeout_s = timeout_s

    def run(self, ctx: RunContext) -> dict[str, Any]:
        model = bind_tools(ctx.chat_model(), [execute_full_solution, mark_solution_complete], ctx.llm)
        task = HumanMessage(ctx.prompt())
        history: list[BaseMessage] = []
        executions = successes = 0
        completed = False

        while self.max_iterations is None or executions < self.max_iterations:
            if executions == 0:
                prompt = [SystemMessage(FIRST_PROMPT), task, HumanMessage(FIRST_ASK)]
            else:
                prompt = [SystemMessage(REFLECT_PROMPT), task, *history]
            response: AIMessage = model.invoke(prompt)
            history.append(response)
            names = [c["name"] for c in response.tool_calls]
            ctx.log("llm", content=response.content, tool_calls=names)
            if not response.tool_calls or "mark_solution_complete" in names:
                completed = "mark_solution_complete" in names
                break
            for call in response.tool_calls:
                if call["name"] != "execute_full_solution":
                    history.append(ToolMessage(f"Unknown tool {call['name']}", tool_call_id=call["id"]))
                    continue
                code = call["args"].get("code", "")
                output = self.execute(code, ctx.run_dir / f"solution_{executions}.py")
                executions += 1
                ok = output.startswith(SUCCESS)
                successes += ok
                if ok:
                    (ctx.run_dir / "solution_final.py").write_text(code)
                history.append(ToolMessage(output, tool_call_id=call["id"], name=call["name"]))
                ctx.log("execution", success=ok, code=code, output=output)

        return {"executions": executions, "successful_executions": successes, "marked_complete": completed}

    def execute(self, code: str, script: Path) -> str:
        """Run a solution in a fresh interpreter and report stdout, stderr and any error."""
        script.write_text(PREAMBLE + code)
        try:
            proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                                  timeout=self.timeout_s, cwd=Path.cwd())
        except subprocess.TimeoutExpired as e:
            return _format(False, _text(e.stdout), _text(e.stderr), f"TimeoutError: exceeded {self.timeout_s}s")
        if proc.returncode == 0:
            return _format(True, proc.stdout, proc.stderr, None)
        lines = proc.stderr.strip().splitlines()
        return _format(False, proc.stdout, "", lines[-1] if lines else f"exit code {proc.returncode}",
                       traceback=proc.stderr)


def _format(ok: bool, stdout: str, stderr: str, error: str | None, traceback: str | None = None) -> str:
    parts = [SUCCESS if ok else FAILURE]
    if stdout:
        parts.append(f"\n--- STDOUT ---\n{stdout}")
    if stderr:
        parts.append(f"\n--- STDERR ---\n{stderr}")
    if error:
        parts.append(f"\n--- ERROR ---\n{error}")
    if traceback:
        parts.append(f"\n--- TRACEBACK ---\n{traceback}")
    return "\n".join(parts)


def _text(data: bytes | str | None) -> str:
    return data.decode(errors="replace") if isinstance(data, bytes) else (data or "")
