#!/usr/bin/env bash
# GGUF pipeline for MenteE Code 3B — run in stages:
#   bash fix_gguf.sh fix      # env + tokenizer surgery + build quantizer
#   bash fix_gguf.sh convert  # F16 -> Q4 -> Q8
#   bash fix_gguf.sh upload   # push to MenteEAI/MenteE-Code-3B-GGUF
set -u
cd /workspace/MenteE-Code-3b
PY=/venv/main/bin/python
PIP=/venv/main/bin/pip

STAGE="${1:-fix}"

if [ "$STAGE" = "fix" ]; then
  echo "== [1] ensure llama.cpp-pinned huggingface_hub (transformers 4.57.6 needs <1.0) =="
  $PIP install "huggingface_hub==0.36.2" -q 2>&1 | grep -E "Successfully|ERROR" | tail -2 || true

  echo "== [2] tokenizer_config surgery: strip extra_special_tokens =="
  $PY - <<'PYEOF'
import json
p = "outputs/merged/tokenizer_config.json"
cfg = json.load(open(p))
if "extra_special_tokens" in cfg:
    cfg.pop("extra_special_tokens", None)
    json.dump(cfg, open(p, "w"), indent=2, ensure_ascii=False)
    print("stripped extra_special_tokens")
else:
    print("extra_special_tokens absent")
PYEOF

  echo "== [3] tokenizer load test (must pass) =="
  $PY - <<'PYEOF'
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("outputs/merged")
print("tokenizer OK, vocab size:", len(t))
PYEOF
  if [ $? -ne 0 ]; then echo "TOKENIZER STILL BROKEN — aborting"; exit 3; fi

  echo "== [4] quantizer binary (build if missing, keep only the binary) =="
  if ! command -v cmake >/dev/null; then
    apt-get update -qq >/dev/null 2>&1 || true
    apt-get install -y -qq cmake g++ >/dev/null 2>&1 || true
  fi
  if [ ! -x ./llama-quantize ]; then
    cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF > /tmp/cmake.log 2>&1
    cmake --build llama.cpp/build --config Release -j"$(nproc)" > /tmp/build.log 2>&1 \
      || { echo "BUILD FAILED — last 20 lines:"; tail -20 /tmp/build.log; exit 4; }
    cp llama.cpp/build/bin/llama-quantize ./llama-quantize
    rm -rf llama.cpp/build
    echo "quantizer built; build objects removed"
  else
    echo "llama-quantize already present"
  fi
  ls -lh ./llama-quantize

  df -h / | tail -1
  echo "STAGE fix DONE"
fi

if [ "$STAGE" = "convert" ]; then
  echo "== [5] F16 master =="
  $PY llama.cpp/convert_hf_to_gguf.py outputs/merged --outfile mentee-f16.gguf --outtype f16 2>&1 | tail -3
  df -h / | tail -1

  echo "== [6] Q4_K_M (the one everyone downloads) =="
  ./llama-quantize mentee-f16.gguf MenteE-Code-3B-Q4_K_M.gguf Q4_K_M 2>&1 | tail -2
  rm -f mentee-f16.gguf
  df -h / | tail -1

  echo "== [7] Q8_0 direct from merged =="
  $PY llama.cpp/convert_hf_to_gguf.py outputs/merged --outfile MenteE-Code-3B-Q8_0.gguf --outtype q8_0 2>&1 | tail -3
  df -h / | tail -1
  ls -lh *.gguf
  echo "STAGE convert DONE"
fi

if [ "$STAGE" = "upload" ]; then
  echo "== [8] publish to MenteEAI/MenteE-Code-3B-GGUF =="
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
  echo "STAGE upload DONE"
fi
