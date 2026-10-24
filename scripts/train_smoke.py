"""Smoke-test training: verify the data pipeline produces trainable rows.

    python scripts/train_smoke.py --data data/external/prepared.jsonl --max-samples 50

Runs a tiny LoRA fine-tune on a small model to confirm the data loads,
tokenizes, and trains without errors. NOT a real training run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tasks/prepared.jsonl")
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--output", default="outputs/smoke_test")
    ap.add_argument("--context-length", type=int, default=32768, help="target context window (32768 native, 131072 with YaRN)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    args = ap.parse_args()

    # Check dependencies
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"  ! missing dependency: {e}")
        print("  install: pip install torch transformers peft trl accelerate bitsandbytes")
        raise SystemExit(1)

    # Load data
    data_path = REPO / args.data
    if not data_path.exists():
        # try generated trajectories
        data_path = REPO / "datasets/collectors/syab/train.jsonl"
    if not data_path.exists():
        print(f"  ! no data at {data_path}")
        raise SystemExit(1)

    rows = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # convert trajectory to a text prompt
        turns = r.get("trajectory", {}).get("turns", [])
        if not turns:
            continue
        text = ""
        for t in turns:
            role = t.get("role", "assistant")
            content = t.get("content", "") or ""
            if content:
                text += f"<|{role}|>\n{content}\n"
        if len(text) > 100:
            rows.append({"text": text})
        if len(rows) >= args.max_samples:
            break

    print(f"  loaded {len(rows)} training samples from {data_path}")
    if len(rows) < 5:
        print("  ! too few samples, need at least 5")
        raise SystemExit(1)

    # Load model
    print(f"  loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rope_scaling = None
    if args.context_length > 32768:
        factor = args.context_length / 32768
        rope_scaling = {"type": "yarn", "factor": factor}
        print(f"  YaRN scaling: 32768 -> {args.context_length} (factor {factor:.1f})")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        rope_scaling=rope_scaling,
    )

    # LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Train
    from datasets import Dataset
    ds = Dataset.from_list(rows)

    out_dir = REPO / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4,
        max_length=args.max_seq_length,
        logging_steps=1,
        save_steps=100,
        bf16=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=config,
    )

    print("  starting smoke test training...")
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"  smoke test complete -> {out_dir}")
    print("  if this ran without errors, your data pipeline is ready for vast.ai")


if __name__ == "__main__":
    raise SystemExit(main())