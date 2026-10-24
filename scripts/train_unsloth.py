"""Unsloth-optimized training for MenteE Code 3B.

    python scripts/train_unsloth.py --data data/training/train.jsonl --max-samples 999999

2-5x faster than standard HuggingFace training. Uses QLoRA by default.
Optimized for RTX 4090 (24GB) on vast.ai.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training/train.jsonl")
    ap.add_argument("--max-samples", type=int, default=999999)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--output", default="outputs/full_run")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--lora-rank", type=int, default=64)
    args = ap.parse_args()

    # Check dependencies
    try:
        import torch
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
    except ImportError as e:
        print(f"  ! missing dependency: {e}")
        print("  install: pip install unsloth torch transformers trl accelerate bitsandbytes")
        raise SystemExit(1)

    # Load data
    data_path = REPO / args.data
    if not data_path.exists():
        print(f"  ! no data at {data_path}")
        raise SystemExit(1)

    rows = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
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

    # Load model with Unsloth (2-5x faster, less VRAM)
    print(f"  loading {args.model} with Unsloth (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    # LoRA config (Unsloth optimized)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    model.print_trainable_parameters()

    # Prepare dataset
    ds = Dataset.from_list(rows)

    out_dir = REPO / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training config
    config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_seq_length,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=config,
    )

    print(f"  training {len(rows)} samples for {args.epochs} epochs...")
    trainer.train()

    # Save merged model (LoRA weights merged into base)
    print("  saving merged model...")
    model.save_pretrained_merged(str(out_dir), tokenizer, save_method="merged_16bit")
    tokenizer.save_pretrained(str(out_dir))

    # Also save GGUF for Ollama
    print("  saving GGUF (Q4_K_M)...")
    model.save_pretrained_gguf(str(out_dir), tokenizer, quantization_method="q4_k_m")

    print(f"  done -> {out_dir}")
    print(f"  merged model: {out_dir}")
    print(f"  GGUF: {out_dir}/*.gguf")
    print("  upload with: python scripts/upload_model.py --model-dir outputs/full_run --repo MenteEAI/MenteE-Code-3B")


if __name__ == "__main__":
    raise SystemExit(main())