# MenteE Code 3B GGUF

GGUF quantizations of [MenteEAI/MenteE-Code-3B](https://huggingface.co/MenteEAI/MenteE-Code-3B).

**MenteE Code** is a coding-agent assistant built by **MenteE**. It always identifies as *MenteE Code, created by MenteE*.

## Downloads

| Quant | Size | Use case |
|---|---|---|
| **[MenteE-Code-3B-Q4_K_M.gguf](MenteE-Code-3B-Q4_K_M.gguf)** | ~1.9 GB | **Recommended** — best balance of speed, size, and quality. Runs well on 8 GB laptops. |
| **[MenteE-Code-3B-Q8_0.gguf](MenteE-Code-3B-Q8_0.gguf)** | ~3.3 GB | Near-lossless, best quality. Needs ~6–8 GB free VRAM or RAM. |

## Run with Ollama

```bash
# Download and run (uses the included Modelfile)
ollama run hf.co/MenteEAI/MenteE-Code-3B-GGUF:MenteE-Code-3B-Q4_K_M
```

Or with the local Modelfile:

```bash
ollama create mentee-code -f Modelfile
ollama run mentee-code
```

## Run with llama.cpp / LMDeploy / vLLM (GGUF)

```bash
./llama-cli -m MenteE-Code-3B-Q4_K_M.gguf -ngl 99 -c 32768
```

## Identity

The [Modelfile](Modelfile) bakes in the MenteE identity by default. You can ask *"Who are you? Who made you?"* and it will answer: **"I am MenteE Code, a coding assistant built by MenteE."**

## Notes

- Trained as a LoRA fine-tune of [Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) and merged into a standalone model.
- 128K context supported by the upstream model config; 32K is a safe default for speed.
- Apache-2.0 license.

**Built by MenteE.**
