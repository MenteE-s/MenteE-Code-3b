from huggingface_hub import HfApi

REPO = "MenteEAI/MenteE-Code-3B-GGUF"
files = HfApi().list_repo_files(REPO)
print("files in", REPO, ":")
for f in files:
    print("  -", f)

required = ["MenteE-Code-3B-Q4_K_M.gguf", "MenteE-Code-3B-Q8_0.gguf", "Modelfile"]
missing = [r for r in required if r not in files]
print("STATUS:", "ALL PRESENT ✓" if not missing else f"MISSING: {missing}")
