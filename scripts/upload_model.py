"""Upload a trained model to HuggingFace Hub (private).

    python scripts/upload_model.py --model-dir outputs/full_run --repo MenteE-s/MenteE-Code-3B

Requires: pip install huggingface_hub
Set HF_TOKEN env var or run: huggingface-cli login
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--repo", required=True, help="HuggingFace repo id, e.g. MenteE-s/MenteE-Code-3B")
    ap.add_argument("--private", action="store_true", default=True)
    args = ap.parse_args()

    model_dir = REPO / args.model_dir
    if not model_dir.exists():
        print(f"  ! model dir not found: {model_dir}")
        raise SystemExit(1)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("  ! set HF_TOKEN env var or run: huggingface-cli login")
        raise SystemExit(1)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("  ! install: pip install huggingface_hub")
        raise SystemExit(1)

    api = HfApi(token=token)
    try:
        create_repo(args.repo, private=args.private, token=token, exist_ok=True)
        print(f"  created/verified repo: {args.repo} (private={args.private})")
    except Exception as e:
        print(f"  ! repo creation: {e}")

    print(f"  uploading {model_dir} -> {args.repo} ...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo,
        repo_type="model",
    )
    print(f"  done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    raise SystemExit(main())