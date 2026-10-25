"""Bake the MenteE identity into the model's chat template.

    python scripts/patch_identity_template.py --model-dir outputs/merged

Training always included the identity system prompt, so the model behaves as
MenteE Code under that condition. This patches chat_template so EVERY client
(transformers, vLLM, llama.cpp, Ollama) automatically sends that prompt even
when the caller provides no system message — without retraining.

The template-level patch is the v1 mechanism; v2 additionally oversamples
identity Q&A rows so the identity lives in the weights themselves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

IDENTITY = (
    "You are MenteE Code, a coding assistant built by MenteE. "
    "You always identify as MenteE Code, created by MenteE."
)

# Jinja2: top-level set inside an `if` persists (if-blocks don't create scope).
INJECT_BLOCK = (
    "{%- set mentee_identity = \"" + IDENTITY + "\" -%}\n"
    "{%- if not (messages and messages[0] is mapping and messages[0].get('role') == 'system') -%}\n"
    "{%- set messages = [{'role': 'system', 'content': mentee_identity}] + messages -%}\n"
    "{%- endif -%}\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="outputs/merged")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"  ! model dir not found: {model_dir}")
        raise SystemExit(1)

    patched = False

    # 1. chat_template.jinja (transformers 5.x / llama.cpp location)
    jinja = model_dir / "chat_template.jinja"
    if jinja.exists():
        text = jinja.read_text(encoding="utf-8")
        if "mentee_identity" in text:
            print("  chat_template.jinja already patched")
        else:
            jinja.write_text(INJECT_BLOCK + text, encoding="utf-8")
            print(f"  patched {jinja}")
        patched = True

    # 2. tokenizer_config.json fallback (older layout)
    tok_cfg = model_dir / "tokenizer_config.json"
    if tok_cfg.exists():
        cfg = json.loads(tok_cfg.read_text(encoding="utf-8"))
        tpl = cfg.get("chat_template")
        if isinstance(tpl, str):
            if "mentee_identity" in tpl:
                print("  tokenizer_config chat_template already patched")
            else:
                cfg["chat_template"] = INJECT_BLOCK + tpl
                tok_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"  patched {tok_cfg} chat_template")
            patched = True

    if not patched:
        print("  ! no chat template found — nothing patched")
        raise SystemExit(1)

    print(f"\n  done. Every conversation now starts as MenteE Code by default.")
    print("  Re-upload so HuggingFace serves the patched files:")
    print(f"    python scripts/upload_model.py --model-dir {model_dir} --repo MenteEAI/MenteE-Code-3B")


if __name__ == "__main__":
    raise SystemExit(main())
