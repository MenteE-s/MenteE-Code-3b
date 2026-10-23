"""Deploy MenteE Code 3B on Modal as a free OpenAI-compatible endpoint.

    modal deploy scripts/modal_deploy.py

Requires: pip install modal
Set up: modal token new
"""
from __future__ import annotations

import modal

app = modal.App("mentee-code-3b")

# Container image with vLLM
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "vllm>=0.6.0",
    "fastapi",
    "uvicorn",
    "huggingface_hub",
)

# Model weights cached in a volume
model_volume = modal.Volume.from_name("mentee-models", create_if_missing=True)
MODEL_PATH = "/models/MenteE-Code-3B"


@app.function(
    gpu="T4",
    image=image,
    volumes={MODEL_PATH: model_volume},
    timeout=600,
    scaledown_window=300,  # scale to zero after 5 min idle
)
@modal.asgi_app()
def serve():
    from vllm.entrypoints.openai.api_server import app as vllm_app
    import os
    os.environ["MODEL_NAME"] = MODEL_PATH
    return vllm_app


@app.function(
    gpu="T4",
    image=image,
    volumes={MODEL_PATH: model_volume},
    timeout=600,
)
def download_model(repo_id: str = "MenteE-s/MenteE-Code-3B"):
    """Download model from HuggingFace to the Modal volume."""
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id, local_dir=MODEL_PATH)
    model_volume.commit()
    print(f"Model downloaded to {MODEL_PATH}")


@app.local_entrypoint()
def main():
    print("Deploying MenteE Code 3B on Modal...")
    print("Endpoint will be available at: https://<your-modal-url>/v1")
    print("Use with OpenCode: base_url = <endpoint>/v1")