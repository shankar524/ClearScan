# ============================================================
# ClearScan OCR - Multi-stage Docker Build (amd64 + arm64)
# ============================================================
# Usage:
#   docker build -t clearscan .
#   docker run --rm -v /path/to/pdfs:/input -v /path/to/results:/output clearscan
# ============================================================

# Pin versions for reproducibility
ARG PYTHON_VERSION=3.11
ARG LLAMA_CPP_VERSION=b8198

# ------------------------------------------------------------
# Stage 1: Obtain llama.cpp binary (platform-aware)
#   amd64 -> download pre-built release (fast)
#   arm64 -> compile from source (no official arm64 release)
# ------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS llama-builder

ARG LLAMA_CPP_VERSION
ARG TARGETARCH

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN set -eux; \
    if [ "$TARGETARCH" = "amd64" ]; then \
        echo ">>> Downloading pre-built llama.cpp for amd64..."; \
        curl -fSL \
            "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_VERSION}/llama-${LLAMA_CPP_VERSION}-bin-ubuntu-x64.tar.gz" \
            -o llama-bin.tar.gz; \
        mkdir -p /build/out; \
        tar -xzf llama-bin.tar.gz -C /build/out --strip-components=1; \
        rm llama-bin.tar.gz; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        echo ">>> Building llama.cpp from source for arm64..."; \
        apt-get update && apt-get install -y --no-install-recommends \
            build-essential cmake git; \
        rm -rf /var/lib/apt/lists/*; \
        git clone --depth 1 --branch ${LLAMA_CPP_VERSION} \
            https://github.com/ggml-org/llama.cpp.git /build/src; \
        cmake -S /build/src -B /build/src/build \
            -DCMAKE_BUILD_TYPE=Release \
            -DGGML_OPENMP=ON \
            -DBUILD_SHARED_LIBS=ON; \
        cmake --build /build/src/build --config Release --target llama-mtmd-cli -j$(nproc); \
        mkdir -p /build/out; \
        cp /build/src/build/bin/llama-mtmd-cli /build/out/; \
        find /build/src/build -name '*.so*' -exec cp {} /build/out/ \; ; \
        rm -rf /build/src; \
    else \
        echo "ERROR: Unsupported architecture: $TARGETARCH"; exit 1; \
    fi; \
    chmod +x /build/out/llama-mtmd-cli; \
    ls -la /build/out/llama-mtmd-cli

# ------------------------------------------------------------
# Stage 2: Download model files (arch-independent)
# ------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS model-downloader

RUN pip install --no-cache-dir huggingface_hub hf_transfer

ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='Qwen/Qwen3-VL-2B-Instruct-GGUF', filename='Qwen3VL-2B-Instruct-Q4_K_M.gguf', local_dir='/models'); \
hf_hub_download(repo_id='Qwen/Qwen3-VL-2B-Instruct-GGUF', filename='mmproj-Qwen3VL-2B-Instruct-F16.gguf', local_dir='/models')"

# ------------------------------------------------------------
# Stage 3: Runtime image
# ------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL maintainer="ClearScan"
LABEL description="ClearScan OCR - PDF-to-text using Qwen3-VL-2B via llama.cpp"

# Runtime system dependencies:
#   libgomp1   - OpenMP (llama.cpp threading)
#   libglib2.0-0 - OpenCV runtime
#   libstdc++6 - C++ stdlib (llama.cpp)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy llama.cpp binary + shared libraries
COPY --from=llama-builder /build/out/ /app/llama_bin/

# Copy model files (~1.6 GB - separate layer for caching)
COPY --from=model-downloader /models/ /app/models/

# Copy application code
COPY run_ocr.py .
COPY download_model.py .
COPY docker-entrypoint.sh .

# Set permissions and register shared libraries
RUN chmod +x /app/llama_bin/llama-mtmd-cli /app/docker-entrypoint.sh \
    && ldconfig /app/llama_bin

# Create volume mount points
RUN mkdir -p /input /output

# Environment defaults for Docker context
ENV CLEARSCAN_LLAMA_CLI=/app/llama_bin/llama-mtmd-cli \
    CLEARSCAN_MODEL_PATH=/app/models/Qwen3VL-2B-Instruct-Q4_K_M.gguf \
    CLEARSCAN_MMPROJ_PATH=/app/models/mmproj-Qwen3VL-2B-Instruct-F16.gguf \
    CLEARSCAN_INPUT_DIR=/input \
    CLEARSCAN_OUTPUT_DIR=/output \
    CLEARSCAN_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/app/llama_bin

ENTRYPOINT ["/app/docker-entrypoint.sh"]
