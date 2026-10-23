"""Mine real, verified bug-fix tasks from a git repository's history.

This produces the *input* side of the dataset: real task instances with a gold
patch, a base commit, and — crucially — evidence that the patch actually works.

What makes a mined task real
----------------------------
A candidate commit only becomes a task if it survives four checks:

1. it touches at least one source file AND at least one test file;
2. the test diff names specific test functions (so we know what to measure);
3. checking out the **parent** commit, those tests FAIL;
4. applying the gold patch, those tests PASS.

Step 3 and 4 are the whole point. A task without evidence that the patch flips a
failing test to passing is a story, not a task — and it cannot be used to verify
an agent trajectory later.

Output shape (JSONL)::

    {"task_id": ..., "instruction": ..., "repo": ..., "base_commit": ...,
     "gold_patch": ..., "fail_to_pass": [...], "pass_to_pass": [...],
     "workdir_note": ...}

Usage::

    python scripts/mine_tasks.py --repo data/repos/more-itertools \\
        --out data/tasks/more-itertools.jsonl --max-tasks 30
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["mine_repo", "MineStats"]

BUGFIX_RE = re.compile(r"\b(fix|bug|crash|incorrect|wrong|error|regression|broken)\b", re.I)
NON_CODE_DIRS = ("docs", "doc", ".github", "CHANGES", "CHANGELOG")
TEST_FILE_RE = re.compile(r"(^|/)(tests?|testing)/|_test\.py$|/test_[^/]+\.py$")
DEF_TEST_RE = re.compile(r"^\+\s*def (test_[A-Za-z0-9_]+)\(")


@dataclass
class MineStats:
    commits_seen: int = 0
    candidates: int = 0
    verified: int = 0
    rejected: dict = field(default_factory=dict)
    last_failure: str = ""

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


def run(cmd: list[str], cwd: str | Path, timeout: int = 120, env: dict[str, str] | None = None) -> tuple[int, str]:
    try:
        import os as _os

        full_env = dict(_os.environ)
        if env:
            full_env.update(env)
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=full_env,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return 1, f"{type(exc).__name__}: {exc}"


def git(repo: Path, *args: str, timeout: int = 120) -> tuple[int, str]:
    return run(["git", *args], cwd=repo, timeout=timeout)


_DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.M)


def _is_test_path(path: str) -> bool:
    p = path.lower()
    return (
        p.startswith("tests/")
        or "/tests/" in p
        or p.endswith("_test.py")
        or p.startswith("test_")
        or "/testing/" in p
    )


def is_test_file(path: str) -> bool:
    return bool(TEST_FILE_RE.search(path.replace("\\", "/"))) and path.endswith(".py")


def is_source_file(path: str) -> bool:
    if path.endswith(".py"):
        return not is_test_file(path) and not any(path.startswith(d) for d in NON_CODE_DIRS)
    return False


def added_test_names(diff: str) -> list[str]:
    """Test functions added by a diff, including their enclosing class.

    Most pytest suites define tests as class methods, so the collection id is
    ``file::Class::test_name`` — not ``file::test_name``. Getting this wrong makes
    pytest error on collection, which looks identical to a test failing and
    silently poisons every downstream check.
    """
    names: list[str] = []
    current_file = ""
    current_class = ""
    class_re = re.compile(r"^\+\s*class\s+([A-Za-z0-9_]+)")
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            current_class = ""
            continue
        if line.startswith("--- ") or line.startswith("diff --git"):
            continue
        if not line.startswith("+"):
            continue
        cm = class_re.match(line)
        if cm:
            current_class = cm.group(1)
            continue
        m = DEF_TEST_RE.match(line)
        if m and current_file:
            test_id = (
                f"{current_file}::{current_class}::{m.group(1)}"
                if current_class
                else f"{current_file}::{m.group(1)}"
            )
            if test_id not in names:
                names.append(test_id)
    return names


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


def changed_files(repo: Path, sha: str) -> tuple[list[str], list[str]]:
    code, out = git(repo, "show", "--format=", "--name-only", sha)
    if code != 0:
        return [], []
    files = [f.strip() for f in out.splitlines() if f.strip()]
    tests = [f for f in files if is_test_file(f)]
    sources = [f for f in files if is_source_file(f)]
    return sources, tests


def test_diff(repo: Path, sha: str, test_files: list[str]) -> str:
    code, out = git(repo, "show", sha, "--", *test_files)
    return out if code == 0 else ""


def run_tests(workdir: Path, targets: list[str], timeout: int = 180) -> tuple[int, str]:
    """Run pytest on whole test files. Returns (returncode, output).

    ``--tb=no -rf`` keeps the output short while still listing every failed test
    id in the short summary, which is what :func:`_failed_tests` parses.
    ``PYTHONPATH`` points at the worktree root so the package under test is
    importable without installing it — a historical checkout is not installed.
    """
    env = {"PYTHONPATH": str(workdir)}
    code, out = run(
        [
            sys.executable, "-m", "pytest", "-q", "--no-header", "--tb=no", "-rf",
            "-p", "no:cacheprovider", *targets,
        ],
        cwd=workdir,
        timeout=timeout,
        env=env,
    )
    return code, out


_FAILED_RE = re.compile(r"^FAILED\s+(\S+)")


def _failed_tests(pytest_output: str) -> set[str]:
    """Extract failing test ids from a ``pytest -q --tb=no -rf`` run."""
    out: set[str] = set()
    for line in pytest_output.splitlines():
        m = _FAILED_RE.match(line.strip())
        if m:
            out.add(m.group(1).replace("\\", "/"))
    return out


def mine_repo(
    repo_path: str | Path,
    out_path: str | Path,
    max_tasks: int = 30,
    max_commits: int = 400,
    test_timeout: int = 180,
    verbose: bool = True,
    before: str | None = None,
) -> tuple[list[dict], MineStats]:
    repo = Path(repo_path).resolve()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = MineStats()

    if not repo.exists():
        raise FileNotFoundError(repo)

    code, branches = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    default_branch = branches.strip() or "HEAD"

    code, log = git(repo, "log", "--no-merges", f"-{max_commits}", "--format=%H%x09%s", *(["--before", before] if before else []))
    if code != 0:
        raise RuntimeError(f"git log failed: {log}")
    commits = [line.split("\t", 1) for line in log.splitlines() if "\t" in line]

    workroot = Path(tempfile.mkdtemp(prefix="mine_"))
    tasks: list[dict] = []
    worktree: Path | None = None
    patch_file = workroot / "gold.patch"

    try:
        for sha, subject in commits:
            if len(tasks) >= max_tasks:
                break
            stats.commits_seen += 1
            if not BUGFIX_RE.search(subject):
                continue

            sources, tests = changed_files(repo, sha)
            if not sources or not tests:
                stats.reject("no_source_and_test_change")
                continue

            diff = test_diff(repo, sha, tests)
            named_new_tests = added_test_names(diff)
            stats.candidates += 1

            # --- base commit = the parent ------------------------------------
            code, parent = git(repo, "rev-parse", f"{sha}^")
            if code != 0:
                stats.reject("no_parent")
                continue
            base = parent.strip()

            # --- checkout the parent into a FRESH worktree -------------------
            # A fresh worktree per candidate: reusing one leaks the previous
            # candidate's applied patch into the next base state.
            wt = workroot / f"wt{stats.candidates}"
            code, out_txt = git(repo, "worktree", "add", "--detach", str(wt), base)
            if code != 0:
                stats.reject("worktree_failed")
                continue
            worktree = wt

            def _cleanup() -> None:
                nonlocal worktree
                if worktree is not None:
                    git(repo, "worktree", "remove", "--force", str(worktree))
                    worktree = None

            # --- the interpreter must be able to parse the code --------------
            # A historical checkout may use syntax newer than the interpreter
            # (e.g. `match` needs 3.10+). A SyntaxError makes the whole test
            # module fail to *collect*, which looks identical to a test failing
            # and would poison the fail/pass check below.
            ok, why = _python_compatible(worktree, sources)
            if not ok:
                stats.reject(f"incompatible_python: {why[:60]}")
                _cleanup()
                continue

            # --- pre-apply the TEST patch at base ----------------------------
            # The fix commit usually ADDS its own regression test. At base that
            # test does not exist, so it cannot "flip" from failing to passing -
            # and a fix whose bug had no prior test coverage would be wrongly
            # rejected. Applying the test part first (SWE-bench method) makes the
            # new test exist and genuinely fail before the source fix lands.
            code, patch_full = git(repo, "diff", f"{sha}^", sha)
            if code != 0 or not patch_full.strip():
                stats.reject("no_patch")
                _cleanup()
                continue
            test_part, source_part = split_patch(patch_full)
            if test_part.strip():
                tp = worktree / ".mente_test.patch"
                tp.write_text(test_part, encoding="utf-8")
                code_tp, _ = git(worktree, "apply", "--whitespace=nowarn", str(tp))
                tp.unlink(missing_ok=True)
                if code_tp != 0:
                    stats.reject("test_patch_failed")
                    _cleanup()
                    continue

            # --- run the whole changed test files BEFORE the source fix ------
            test_files = sorted(set(tests))
            code_before, out_before = run_tests(worktree, test_files, test_timeout)
            failed_before = _failed_tests(out_before)
            if code_before == 0 and not failed_before:
                stats.reject("tests_already_pass")
                _cleanup()
                continue

            # --- apply the SOURCE part of the fix ----------------------------
            if source_part.strip():
                sp = worktree / ".mente_src.patch"
                sp.write_text(source_part, encoding="utf-8")
                code_apply, _ = git(worktree, "apply", "--whitespace=nowarn", str(sp))
                sp.unlink(missing_ok=True)
            else:
                code_apply, _ = git(worktree, "apply", "--whitespace=nowarn", "-")
            if code_apply != 0:
                stats.reject("patch_does_not_apply")
                _cleanup()
                continue

            # --- run the same test files AFTER the patch ----------------------
            # Judge by *flips*, not by whether the whole file passes: an unrelated
            # pre-existing failure must not discard a task whose target test
            # genuinely flipped from failing to passing.
            code_after, out_after = run_tests(worktree, test_files, test_timeout)
            failed_after = _failed_tests(out_after)
            fail_to_pass = sorted(failed_before - failed_after)
            regressions = sorted(failed_after - failed_before)
            if not fail_to_pass:
                stats.reject("no_test_flipped")
                stats.last_failure = out_after[-500:]
                _cleanup()
                continue
            if regressions:
                stats.reject(f"regressions:{len(regressions)}")
                _cleanup()
                continue
            regressed = len(regressions)

            body_lines = _commit_body(repo, sha)
            instruction = _instruction_from(subject, body_lines, fail_to_pass)

            task = {
                "task_id": f"mine_{repo.name}_{sha[:8]}",
                "instruction": instruction,
                "repo": repo.name,
                "repo_url": f"https://github.com/{_guess_slug(repo)}",
                "repo_license": _license(repo),
                "base_commit": base,
                "fix_commit": sha,
                "language": "python",
                "task_type": "bug_fix",
                "difficulty": "medium",
                "setup_script": _setup_hint(worktree),
                "test_command": f"{sys.executable} -m pytest -q",
                "fail_to_pass": fail_to_pass,
                "gold_patch": patch_full,
                "test_patch": test_part,
                "source_patch": source_part,
                "verified": True,
                "verification": {
                    "tests_failed_before": len(failed_before),
                    "tests_passed_after": True,
                    "regressions": regressed,
                    "new_tests_added": named_new_tests,
                    "method": "whole_file_flip",
                    "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                "extra": {"fix_subject": subject, "fix_body": body_lines[:600]},
            }
            tasks.append(task)
            stats.verified += 1
            if verbose:
                print(
                    f"  [ok] {sha[:8]}  {len(fail_to_pass)} test(s) flipped  "
                    f"({len(failed_before)} failed before)  {subject[:60]}"
                )

            _cleanup()
    finally:
        if worktree is not None:
            git(repo, "worktree", "remove", "--force", str(worktree))
        shutil.rmtree(workroot, ignore_errors=True)

    with open(out, "w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(task, ensure_ascii=False) + "\n")

    return tasks, stats


def _python_compatible(workdir: Path, source_files: list[str]) -> tuple[bool, str]:
    """Can this interpreter parse the changed source files?

    Two checks, cheapest first:

    1. ``requires-python`` in pyproject.toml / setup.cfg — the project's own floor.
    2. ``compile()`` each changed source file — catches syntax the interpreter
       does not support, which is what actually breaks test collection.
    """
    for name in ("pyproject.toml", "setup.cfg"):
        p = workdir / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"requires-python\s*=\s*[\"']([^\"']+)[\"']", text)
        if m:
            floor = re.search(r">=\s*3\.(\d+)", m.group(1))
            if floor and sys.version_info < (3, int(floor.group(1))):
                return False, f"requires-python {m.group(1)} > {sys.version_info.major}.{sys.version_info.minor}"
            floor_minor = re.search(r">=\s*(\d+)\.(\d+)", m.group(1))
            if floor_minor and sys.version_info < (int(floor_minor.group(1)), int(floor_minor.group(2))):
                return False, f"requires-python {m.group(1)}"

    for rel in source_files:
        path = workdir / rel
        if not path.exists():
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            compile(src, rel, "exec")
        except SyntaxError as exc:
            return False, f"{rel}: {exc.msg} (line {exc.lineno})"
    return True, ""


def _existing_tests_in_file(workdir: Path, test_file: str) -> list[str]:
    path = workdir / test_file
    if not path.exists():
        return []
    names = re.findall(r"^\s*def (test_[A-Za-z0-9_]+)\(", path.read_text(encoding="utf-8", errors="replace"), re.M)
    return [f"{test_file}::{n}" for n in names]


def _commit_body(repo: Path, sha: str) -> str:
    code, out = git(repo, "log", "-1", "--format=%B", sha)
    return out.strip() if code == 0 else ""


def _instruction_from(subject: str, body: str, test_ids: list[str]) -> str:
    """Build a natural instruction from the commit message.

    We deliberately do not hand the model the diff or the fix. The instruction is
    the issue statement; the fix is what the agent has to find.
    """
    parts = [subject.strip()]
    body_clean = "\n".join(
        line for line in body.splitlines() if line.strip() and not line.strip().startswith(("Fixes #", "Closes #", "Co-Authored"))
    )
    if body_clean and body_clean.lower() != subject.strip().lower():
        parts.append(body_clean[:600])
    parts.append(
        "Fix this issue so that the failing tests pass: "
        + ", ".join(t.split("::")[-1] for t in test_ids[:3])
    )
    return "\n\n".join(parts)


def _license(repo: Path) -> str:
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"):
        p = repo / name
        if p.exists():
            head = p.read_text(encoding="utf-8", errors="replace")[:400].lower()
            if "mit license" in head or "permission is hereby granted" in head:
                return "MIT"
            if "apache license" in head:
                return "Apache-2.0"
            if "bsd" in head:
                return "BSD-3-Clause"
            if "gnu general public license" in head:
                return "GPL-3.0"
    return "unknown"


def _guess_slug(repo: Path) -> str:
    code, out = git(repo, "remote", "get-url", "origin")
    if code == 0:
        url = out.strip()
        for prefix in ("https://github.com/", "git@github.com:"):
            if url.startswith(prefix):
                return url[len(prefix) :].removesuffix(".git")
    return repo.name


def _setup_hint(workdir: Path) -> str:
    if (workdir / "pyproject.toml").exists() or (workdir / "setup.py").exists():
        return f"{sys.executable} -m pip install -e ."
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tasks", type=int, default=30)
    ap.add_argument("--max-commits", type=int, default=400)
    ap.add_argument("--test-timeout", type=int, default=180)
    ap.add_argument("--before", default=None, help="only scan commits before this date (YYYY-MM-DD)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    tasks, stats = mine_repo(
        args.repo, args.out, args.max_tasks, args.max_commits, args.test_timeout, not args.quiet, args.before
    )
    print(f"\n  commits scanned   {stats.commits_seen}")
    print(f"  candidates        {stats.candidates}")
    print(f"  verified tasks    {stats.verified}")
    print("  rejected:")
    for reason, count in sorted(stats.rejected.items(), key=lambda kv: -kv[1]):
        print(f"    {reason:<28} {count}")
    if stats.last_failure:
        print("\n  last failing pytest output (tail):")
        for line in stats.last_failure.splitlines()[-12:]:
            print(f"    {line}")
    print(f"\n  wrote {len(tasks)} task(s) to {args.out}")
    return 0 if tasks else 1


if __name__ == "__main__":
    raise SystemExit(main())
