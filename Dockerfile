# syntax=docker/dockerfile:1.14
# (1.14: COPY --chmod accepts symbolic modes such as u=rwX,go=rX — used for the models below)
#
# TinyGPT service: the API + the Gradio demo, both models inside, CPU only.
#
#   python scripts/export.py                 # models/ must exist first
#   docker build -t tinygpt .
#   docker run --rm -p 8000:8000 tinygpt     # http://localhost:8000 (demo), /docs (API)
#
# Settings are environment variables: docker run -e TINYGPT_MAX_BATCH=1 ...  (see serve/api.py)

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# Layers from least to most often changed, so a code change rebuilds only the last few.
# 1. PyTorch, CPU-only build: the default Linux wheel brings CUDA libraries (several GB) this
#    image would never use.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1

# 2. Everything else, at the exact versions in the lock file.
COPY requirements/serve.txt requirements/serve.txt
RUN pip install -r requirements/serve.txt

# 3. Our code, then the exported models (scripts/export.py).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .
# --chmod: COPY keeps each file's permissions from your disk, and safetensors writes weights as
# rw------- (owner only). Copied as-is, root would own them and the app user below couldn't read
# them. u=rwX,go=rX: everyone may read, directories stay enterable, only root may change anything.
COPY --chmod=u=rwX,go=rX models ./models

# Never run as root: if the app is ever compromised, the attacker is a user who owns nothing.
RUN useradd --create-home --uid 1000 app
USER app

ENV TINYGPT_LM=/app/models/lm \
    TINYGPT_PARSER=/app/models/parser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["python", "-m", "tinygpt.serve", "--host", "0.0.0.0", "--port", "8000"]
