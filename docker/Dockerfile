# syntax=docker/dockerfile:1.7
# DefectLens: single image that serves the API and the built web UI on port 8000.
#
#   docker compose -f docker/docker-compose.yml up --build                  # CPU
#   docker build -f docker/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 .   # CUDA
#
# NOTE: this Dockerfile was written but could NOT be built in the project's sandbox (no Docker daemon).
#       Every step mirrors the native install that was tested; see README "Docker".

########## 1) build the web UI ##########
FROM node:20-bookworm-slim AS frontend
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json frontend/.npmrc ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

########## 2) runtime ##########
FROM python:3.13-slim-bookworm AS runtime

# CPU wheels by default (about 200 MB). For NVIDIA GPUs pass the matching CUDA index, e.g. .../whl/cu126.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DL_VAR_DIR=/data \
    TORCH_HOME=/data/torch \
    DL_SERVE_FRONTEND=true \
    DL_LOG_FORMAT=json

# libglib2.0-0: required by opencv-python-headless; tini: proper signal handling / zombie reaping
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
RUN pip install --index-url "${TORCH_INDEX}" "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
COPY backend/requirements.lock ./
RUN pip install -r requirements.lock
COPY backend/ ./
RUN pip install --no-deps -e .
COPY --from=frontend /src/frontend/dist /app/frontend/dist
COPY samples/ /app/samples/

RUN useradd --system --uid 10001 --home-dir /data defectlens \
 && mkdir -p /data && chown -R defectlens /data
USER defectlens
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health/ready', timeout=4).status == 200 else 1)"

ENTRYPOINT ["tini", "--"]
# One worker on purpose: rate limits and the in-process job worker are per-process (see SECURITY.md).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
