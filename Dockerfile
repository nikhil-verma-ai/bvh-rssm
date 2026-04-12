# BVH-RSSM Serving Container
#
# Installs only runtime serving dependencies (torch, fastapi, pydantic, uvicorn).
# Training deps (wandb, hydra, pycox, mujoco, minigrid) are NOT installed.
#
# Build:
#   docker build -t bvh-rssm-serving:latest .
#
# Run (with checkpoint mounted):
#   docker run --rm -p 8000:8000 \
#     -v /path/to/runs:/runs:ro \
#     bvh-rssm-serving:latest \
#     --checkpoint /runs/phase2/step10000.pt
#
# Run (fast mode, no checkpoint):
#   docker run --rm -p 8000:8000 bvh-rssm-serving:latest --fast-mode

FROM python:3.11-slim

# Install system deps required by torch CPU wheel (no CUDA)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./

# Install serving dependencies before copying source so this layer is
# cached across code changes. torch is ~2 GB — re-downloading on every
# bvh_rssm/ edit defeats the point of layer caching.
RUN pip install --no-cache-dir \
    "torch>=2.0" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.29" \
    "pydantic>=2.6" \
    "numpy>=1.26"

COPY bvh_rssm/ ./bvh_rssm/
COPY scripts/serve.py ./scripts/serve.py

RUN pip install --no-cache-dir -e . --no-deps

EXPOSE 8000

# Default: serve on 0.0.0.0:8000 — caller must provide --checkpoint or --fast-mode
ENTRYPOINT ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
CMD []
