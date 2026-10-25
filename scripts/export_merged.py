"""Merge the trained LoRA adapter into the base model -> standalone model.

    python scripts/export_merged.py \
      --adapter outputs/full_run \
      --base Qwen/Qwen2.5-Coder-3B-Instruct \
      --out outputs/merged \
      --context-length 131072

PEFT training saves only the small adapter; this bakes it into a complete
model (config.json + weights + tokenizer) that anyone can download and run,
and applies the YaRN rope-scaling patch for extended context.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/full_run", help="dir with adapter_model.safetensors")
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--out", default="outputs/merged")
    ap.add_argument("--context-length", type=int, default=131072)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"  loading base {args.base} (CPU, bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    print(f"  loading adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("  merging LoRA into base weights...")
    model = model.merge_and_unload()

    print(f"  saving -> {out}")
    model.save_pretrained(out)

    tok_src = args.adapter if (Path(args.adapter) / "tokenizer_config.json").exists() else args.base
    print(f"  saving tokenizer from {tok_src}...")
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(out)

    # YaRN rope scaling for extended context (transformers 5.x reads rope_type)
    if args.context_length > 32768:
        cfg_path = out / "config.json"
        cfg = json.loads(cfg_path.read_text())
        factor = args.context_length / 32768
        cfg["rope_scaling"] = {
            "rope_type": "yarn",
            "type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": 32768,
        }
        cfg["max_position_embeddings"] = args.context_length
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"  YaRN scaling applied: 32768 -> {args.context_length} (factor {factor:.1f})")

    print(f"  done -> {out}")
    print("  upload with:")
    print(f"    python scripts/upload_model.py --model-dir {out} --repo MenteEAI/MenteE-Code-3B")


if __name__ == "__main__":
    raise SystemExit(main())
