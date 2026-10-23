"""Mine bug-fix tasks from many repos in one pass.

    python scripts/mine_all_repos.py --repos data/repos_to_mine.txt --out data/tasks/all_tasks.jsonl

Clones each repo (if not already cloned), runs the miner, and aggregates
all verified tasks into a single file. Skips repos that fail or are too large.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv312" / "Scripts" / "python.exe"


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, **kw)


def clone_repo(url: str, name: str, repos_root: Path) -> Path | None:
    repo_dir = repos_root / name
    if repo_dir.exists() and (repo_dir / ".git").exists():
        print(f"  [skip clone] {name} already exists")
        return repo_dir
    print(f"  cloning {name}...")
    # shallow clone first to check size
    code, out = run(
        ["git", "clone", "--depth", "1", url, str(repo_dir)],
        timeout=300,
    )
    if code != 0:
        print(f"  ! clone failed: {out[:200]}")
        return None
    # unshallow for git history mining
    run(["git", "fetch", "--unshallow"], cwd=repo_dir, timeout=600)
    return repo_dir


def mine_repo(repo_dir: Path, name: str, out_file: Path, max_tasks: int, max_commits: int):
    repo_rel = repo_dir.relative_to(REPO).as_posix()
    tmp_out = REPO / "data" / "tasks" / f"_tmp_{name}.jsonl"
    print(f"  mining {name}...")
    code, out = run(
        [
            str(PY), "scripts/mine_tasks.py",
            "--repo", repo_rel,
            "--out", str(tmp_out.relative_to(REPO)),
            "--max-tasks", str(max_tasks),
            "--max-commits", str(max_commits),
            "--test-timeout", "200",
        ],
        timeout=3600,
    )
    # extract summary line
    for line in out.splitlines():
        if "verified tasks" in line or "wrote" in line:
            print(f"    {line.strip()}")
    if tmp_out.exists() and tmp_out.stat().st_size > 0:
        # append to aggregate
        with out_file.open("a", encoding="utf-8") as dst:
            for line in tmp_out.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    dst.write(line + "\n")
        tmp_out.unlink()
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default="data/repos_to_mine.txt")
    ap.add_argument("--out", default="data/tasks/all_tasks.jsonl")
    ap.add_argument("--repos-root", default="data/repos")
    ap.add_argument("--max-tasks", type=int, default=20, help="max tasks per repo")
    ap.add_argument("--max-commits", type=int, default=500, help="max commits to scan per repo")
    ap.add_argument("--skip-clone", action="store_true", help="don't clone, use existing repos")
    ap.add_argument("--only", default=None, help="comma-separated repo names to process")
    args = ap.parse_args()

    repos_root = REPO / args.repos_root
    repos_root.mkdir(parents=True, exist_ok=True)
    out_file = REPO / args.out
    # start fresh
    if out_file.exists():
        out_file.unlink()

    # parse repos list
    repos = []
    for line in (REPO / args.repos).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            repos.append((parts[0], parts[1]))

    if args.only:
        wanted = set(args.only.split(","))
        repos = [r for r in repos if r[1] in wanted]

    print(f"Mining {len(repos)} repos...\n")
    t0 = time.time()
    total_tasks = 0
    for i, (url, name) in enumerate(repos, 1):
        print(f"[{i}/{len(repos)}] {name}")
        if not args.skip_clone:
            repo_dir = clone_repo(url, name, repos_root)
        else:
            repo_dir = repos_root / name
        if repo_dir is None or not repo_dir.exists():
            print(f"  ! skipping {name} (no repo)")
            continue
        # check size
        size_mb = sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file()) / 1e6
        if size_mb > 2000:
            print(f"  ! skipping {name} (too large: {size_mb:.0f}MB)")
            continue
        ok = mine_repo(repo_dir, name, out_file, args.max_tasks, args.max_commits)
        if ok:
            # count new tasks
            n = sum(1 for _ in out_file.read_text(encoding="utf-8").splitlines())
            print(f"  -> total tasks so far: {n}")
            total_tasks = n
        print()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min. Total tasks: {total_tasks}")
    print(f"Output: {out_file}")
    print(f"\nNext: python scripts/prepare_tasks.py --in {out_file} --out data/tasks/prepared_all.jsonl")


if __name__ == "__main__":
    raise SystemExit(main())