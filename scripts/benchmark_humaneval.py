"""Honest benchmark: HumanEval pass@1, measured — not estimated.

    python scripts/benchmark_humaneval.py --model outputs/full_run --limit 50
    python scripts/benchmark_humaneval.py --model Qwen/Qwen2.5-Coder-3B-Instruct --limit 50

Runs the model on the published HumanEval problems (OpenAI, 164 Python tasks),
executes each generated solution against the bundled unit tests in a subprocess
with a timeout, and reports the real pass@1.

Run it on the SAME machine for both the base model and the tuned model to get
a fair head-to-head. Requires only transformers + torch (no vLLM).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def extract_code(text: str) -> str:
    """Pull the python code block out of a chat answer, or use the raw text."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    # no fences: take everything up to the first markdown artifact
    return text.split("```")[0].strip()


def run_one(model, tokenizer, problem: dict, max_new_tokens: int) -> tuple[bool, str]:
    """Generate a solution for one problem and execute its tests. Returns (passed, log)."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are MenteE Code, a coding assistant built by MenteE. "
                "Write correct, complete Python code."
            ),
        },
        {
            "role": "user",
            "content": (
                "Complete the following Python function. Reply with the complete "
                "function in a single ```python code block, no explanations.\n\n"
                + problem["prompt"]
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    import torch

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: deterministic, standard for pass@1
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    code = extract_code(answer)

    # HumanEval harness: solution + test + the canonical check() call
    harness = code + "\n\n" + problem["test"] + f"\ncheck({problem['entry_point']})\n"

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "solution.py"
        path.write_text(harness, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=td,
            )
            passed = proc.returncode == 0
            log = (proc.stdout + proc.stderr)[-800:]
        except subprocess.TimeoutExpired:
            passed, log = False, "TIMEOUT"

    return passed, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local path or HF id")
    ap.add_argument("--limit", type=int, default=164, help="problems to evaluate (max 164)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--output", default="outputs/humaneval_results.json")
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"  loading HumanEval (164 problems, running {args.limit})...")
    ds = list(load_dataset("openai_humaneval", split="test"))[: args.limit]

    print(f"  loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    model.eval()

    passed = 0
    runs = []
    t0 = time.time()
    for i, problem in enumerate(ds, 1):
        ok, log = run_one(model, tokenizer, problem, args.max_new_tokens)
        passed += int(ok)
        runs.append(
            {
                "task_id": problem["task_id"],
                "passed": ok,
                "log": "" if ok else log,
            }
        )
        print(
            f"  [{i}/{len(ds)}] {problem['task_id']}  {'PASS' if ok else 'FAIL'}"
            f"  ({passed}/{i} = {100 * passed / i:.1f}%)"
        )
        # incremental save so a crash keeps partial results
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(
                {
                    "model": args.model,
                    "benchmark": "HumanEval",
                    "n": i,
                    "passed": passed,
                    "pass_at_1": round(100 * passed / i, 2),
                    "runs": runs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    mins = (time.time() - t0) / 60
    print(f"\n{'=' * 60}")
    print(f"  HumanEval pass@1: {passed}/{len(ds)} = {100 * passed / len(ds):.1f}%")
    print(f"  model: {args.model}")
    print(f"  time: {mins:.1f} min   results: {args.output}")
    print(f"{'=' * 60}")
    print("\n  published pass@1 for context (vendor-reported, approximate):")
    print("    Claude 3.5 Sonnet ~92 | GPT-4o ~91 | DeepSeek V3 ~89")
    print("    Qwen2.5-Coder-32B ~92 | Qwen2.5-Coder-3B ~84")
    print("    -> compare YOUR two numbers (base vs tuned) head-to-head.")


if __name__ == "__main__":
    raise SystemExit(main())
