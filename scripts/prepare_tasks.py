"""Prepare mined tasks for the agent: worktree + test patch + verification.

A mined task stores a gold patch and a base commit, but the agent cannot run on
that directly. For each task this script:

1. creates a worktree at the task's ``base_commit``;
2. splits the gold patch into its **test** part and its **source** part;
3. applies only the TEST part, so the regression test exists and fails;
4. verifies the failing tests really do fail in that worktree;
5. writes a *prepared* task with ``workdir`` filled in.

The source part is kept alongside so the eventual merge can check the agent's
patch against the reference fix - it is never shown to the agent.

Verification is the point: a task whose tests do not fail in the prepared workdir
cannot tell a correct fix from a wrong one, so it is dropped here rather than
wasting an API call later.

Usage::

    python scripts/prepare_tasks.py --in data/tasks/more-itertools.jsonl \\
        --out data/tasks/prepared.jsonl [--limit 30]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.mine_tasks import run  # noqa: E402  (same subprocess helper)

_DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.M)


def split_patch(gold: str) -> tuple[str, str]:
    """Split a unified diff into (test part, source part)."""
    if not gold.strip():
        return "", ""
    matches = list(_DIFF_HEADER.finditer(gold))
    test_parts: list[str] = []
    source_parts: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(gold)
        section = gold[start:end]
        b_path = m.group(2).replace("\\", "/")
        if _is_test_path(b_path):
            test_parts.append(section)
        else:
            source_parts.append(section)
    return "\n".join(test_parts), "\n".join(source_parts)


def _is_test_path(path: str) -> bool:
    p = path.lower()
    return p.startswith("tests/") or "/tests/" in p or p.endswith("_test.py") or p.startswith("test_")


def prepare_one(task: dict, repos_root: Path, workdirs_root: Path, py: str, timeout: int) -> dict | None:
    repo_dir = repos_root / task["repo"]
    if not repo_dir.exists():
        return None

    base = task.get("base_commit") or ""
    if not base:
        return None

    task = task.copy()
    # worktrees are created via git with cwd=repo_dir, so relative paths would
    # resolve inside the repo - make everything absolute up front.
    workdirs_root.mkdir(parents=True, exist_ok=True)
    wt = (workdirs_root / task["task_id"]).resolve()
    if wt.exists():
        run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo_dir, timeout=300)
    code, out = run(["git", "worktree", "add", "--force", "--detach", str(wt), base], cwd=repo_dir, timeout=300)
    if code != 0:
        return None

    test_patch, source_patch = split_patch(task.get("gold_patch", ""))

    if test_patch.strip():
        tp = wt / ".mente_test.patch"
        tp.write_text(test_patch, encoding="utf-8")
        code, out = run(["git", "apply", "--whitespace=nowarn", str(tp)], cwd=wt, timeout=120)
        tp.unlink(missing_ok=True)
        if code != 0:
            run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo_dir)
            return None

    # save the source part for later comparison
    if source_patch.strip():
        side = workdirs_root / f"{task['task_id']}.source.patch"
        side.write_text(source_patch, encoding="utf-8")

    # the failing tests must actually fail in this worktree
    f2p = task.get("fail_to_pass") or []
    env = {"PYTHONPATH": str(wt)}
    if f2p:
        code, out = run(
            [py, "-m", "pytest", "-q", "--no-header", "--tb=no", "-rf", *f2p],
            cwd=wt, timeout=timeout, env=env,
        )
        if code == 0:
            # passes without a fix -> not a bug; drop
            run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo_dir)
            return None

    prepared = dict(task)
    prepared["workdir"] = str(wt)
    prepared["test_patch"] = test_patch
    prepared["source_patch"] = source_patch
    prepared["prepared_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return prepared


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repos-root", default="data/repos")
    ap.add_argument("--workdirs-root", default="data/workdirs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--test-timeout", type=int, default=240)
    args = ap.parse_args()

    src = Path(args.inp)
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    workdirs_root = Path(args.workdirs_root)
    workdirs_root.mkdir(parents=True, exist_ok=True)

    py = str(Path(__file__).resolve().parents[1] / ".venv312" / "Scripts" / "python.exe")
    if not Path(py).exists():
        py = sys.executable

    ok = 0
    with open(out, "w", encoding="utf-8") as fh:
        for i, task in enumerate(rows, 1):
            prepared = prepare_one(
                task, Path(args.repos_root), workdirs_root, py, args.test_timeout
            )
            if prepared is None:
                print(f"  [{i}/{len(rows)}] SKIP  {task.get('task_id')}")
                continue
            fh.write(json.dumps(prepared, ensure_ascii=False) + "\n")
            ok += 1
            print(f"  [{i}/{len(rows)}] OK    {task.get('task_id')}  ({len(prepared.get('fail_to_pass', []))} failing test(s))")

    print(f"\n  prepared {ok}/{len(rows)} tasks -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
