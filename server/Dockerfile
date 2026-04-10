# Copyright (c) 2026 CodeBlockers. All rights reserved.

# Multi-stage build for IncidentOps OpenEnv environment.
# Compatible with both the openenv-base image and a plain Python image.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

ARG BUILD_MODE=in-repo

# Copy the entire environment directory into the container
COPY . /app/env

WORKDIR /app/env

# Ensure uv is available (may already be present in openenv-base)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install git for git-based dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Layer 1: install dependencies only (for Docker layer caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

# Layer 2: install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# ── Final runtime stage ──────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/env/.venv /app/.venv

# Copy the application code
COPY --from=builder /app/env /app/env

# Activate venv
ENV PATH="/app/.venv/bin:$PATH"

# Make all modules importable from /app/env
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check — uses Python stdlib so no curl/wget required
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose the API port
EXPOSE 8000

# Start the FastAPI server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
