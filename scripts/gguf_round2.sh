#!/usr/bin/env bash
# GGUF round 2 — canonical order with real space checks
#   bash gguf_round2.sh convert   # restore merged -> F16 -> Q4 -> Q8
#   bash gguf_round2.sh upload    # publish to MenteEAI/MenteE-Code-3B-GGUF
set -u
cd /workspace/MenteE-Code-3b
PY=/venv/main/bin/python

if [ "${1:-convert}" = "convert" ]; then
  echo "== [1] restore merged model from HuggingFace =="
  $PY - <<'PYEOF'
import json, pathlib
from huggingface_hub import snapshot_download
snapshot_download("MenteEAI/MenteE-Code-3B", local_dir="outputs/merged")
print("restored")
p = pathlib.Path("outputs/merged/tokenizer_config.json")
if p.exists():
    cfg = json.loads(p.read_text())
    if "extra_special_tokens" in cfg:
        cfg.pop("extra_special_tokens")
        p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        print("stripped extra_special_tokens")
PYEOF
  df -h / | tail -1

  echo "== [2] drop old quants (rebuilding both from F16) =="
  rm -f MenteE-Code-3B-Q8_0.gguf MenteE-Code-3B-Q4_K_M.gguf
  df -h / | tail -1

  echo "== [3] F16 master (needs ~6.2G free) =="
  $PY llama.cpp/convert_hf_to_gguf.py outputs/merged --outfile mentee-f16.gguf --outtype f16 > /tmp/f16.log 2>&1
  SIZE=$(stat -c%s mentee-f16.gguf 2>/dev/null || echo 0)
  if [ "$SIZE" -lt 5000000000 ]; then echo "F16 FAILED (size $SIZE)"; tail -5 /tmp/f16.log; exit 5; fi
  ls -lh mentee-f16.gguf
  df -h / | tail -1

  echo "== [4] Q4_K_M from F16 =="
  ./llama-quantize mentee-f16.gguf MenteE-Code-3B-Q4_K_M.gguf Q4_K_M > /tmp/q4.log 2>&1 \
    || { echo "Q4 FAILED"; tail -5 /tmp/q4.log; exit 6; }
  rm -f mentee-f16.gguf
  df -h / | tail -1

  echo "== [5] Q8_0 from merged =="
  $PY llama.cpp/convert_hf_to_gguf.py outputs/merged --outfile MenteE-Code-3B-Q8_0.gguf --outtype q8_0 > /tmp/q8.log 2>&1 \
    || { echo "Q8 FAILED"; tail -5 /tmp/q8.log; exit 7; }
  rm -rf outputs/merged
  ls -lh *.gguf
  df -h / | tail -1
  echo "CONVERT2 DONE"
fi

if [ "${1:-convert}" = "upload" ]; then
  echo "== [6] publish =="
  $PY - <<'PYEOF'
from huggingface_hub import HfApi
api = HfApi()
repo = "MenteEAI/MenteE-Code-3B-GGUF"
api.create_repo(repo, private=True, exist_ok=True, repo_type="model")
for f in ["MenteE-Code-3B-Q4_K_M.gguf", "MenteE-Code-3B-Q8_0.gguf"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=repo)
    print("uploaded:", f)
open("Modelfile", "w").write(
    "FROM ./MenteE-Code-3B-Q4_K_M.gguf\n"
    "SYSTEM \"\"\"You are MenteE Code, a coding assistant built by MenteE. "
    "You always identify as MenteE Code, created by MenteE.\"\"\"\n"
    "PARAMETER temperature 0.2\nPARAMETER num_ctx 32768\n"
)
api.upload_file(path_or_fileobj="Modelfile", path_in_repo="Modelfile", repo_id=repo)
print("GGUF REPO COMPLETE")
PYEOF
  echo "UPLOAD DONE"
fi
