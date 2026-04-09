#!/bin/bash
#
# PhantomRAM — reproduction setup script.
#
# Builds PhantomRAM, runs its unit tests, clones and builds llama.cpp,
# and optionally downloads a test model for benchmarking.
#
# Usage:
#   ./scripts/setup.sh                       # Build PhantomRAM + llama.cpp
#   ./scripts/setup.sh --model               # Also download Llama 3.1 8B (needs HF CLI)
#   LLAMA_REF=b4000 ./scripts/setup.sh       # Pin llama.cpp to a specific tag/SHA
#

set -euo pipefail

# llama.cpp reference to check out. The v0.3 benchmarks were run against
# llama.cpp master as of 2026-04-08, but no exact commit was recorded at
# the time. Override with LLAMA_REF=<tag-or-sha> ./scripts/setup.sh to
# pin to a specific version for reproducible runs.
LLAMA_REF="${LLAMA_REF:-master}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> PhantomRAM setup"
echo "    repo root: ${REPO_ROOT}"

# ── Step 1: build PhantomRAM and run its tests ───────────────────────────
echo ""
echo "==> Building libphantom_preload.so and running tests"
cd "${REPO_ROOT}"
make clean
make
make test

# ── Step 2: clone llama.cpp if missing ───────────────────────────────────
if [ ! -d "${REPO_ROOT}/llama.cpp" ]; then
    echo ""
    echo "==> Cloning llama.cpp @ ${LLAMA_REF}"
    git clone https://github.com/ggerganov/llama.cpp.git "${REPO_ROOT}/llama.cpp"
    cd "${REPO_ROOT}/llama.cpp"
    git checkout "${LLAMA_REF}"
else
    echo ""
    echo "==> llama.cpp already present — skipping clone"
    cd "${REPO_ROOT}/llama.cpp"
fi

# ── Step 3: build llama.cpp ──────────────────────────────────────────────
echo ""
echo "==> Building llama.cpp (CPU only, no CUDA)"
if [ ! -f build/bin/llama-simple ]; then
    cmake -B build -DGGML_CUDA=OFF -DGGML_NATIVE=ON
    cmake --build build --config Release -j --target llama-simple llama-cli
fi

# ── Step 4: optional model download ──────────────────────────────────────
if [ "${1:-}" = "--model" ]; then
    MODEL_DIR="${REPO_ROOT}/models"
    mkdir -p "${MODEL_DIR}"
    MODEL_FILE="${MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    if [ ! -f "${MODEL_FILE}" ]; then
        echo ""
        echo "==> Downloading Llama 3.1 8B Instruct Q4_K_M (~4.6 GB)"
        echo "    (requires huggingface-cli login with a gated-model token)"
        if command -v huggingface-cli >/dev/null 2>&1; then
            huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
                Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
                --local-dir "${MODEL_DIR}" --local-dir-use-symlinks False
        else
            echo "    huggingface-cli not installed — skipping."
            echo "    Install: pip install huggingface_hub[cli]"
        fi
    fi
fi

echo ""
echo "==> Setup complete."
echo ""
echo "    Try the benchmark:"
echo "      sync && echo 3 | sudo tee /proc/sys/vm/drop_caches"
echo "      ./build/mmap_bench /path/to/model.gguf"
echo ""
echo "      sync && echo 3 | sudo tee /proc/sys/vm/drop_caches"
echo "      LD_PRELOAD=./build/libphantom_preload.so PHANTOM_VERBOSE=1 \\"
echo "          ./build/mmap_bench /path/to/model.gguf"
echo ""
echo "    Try with llama.cpp:"
echo "      LD_PRELOAD=./build/libphantom_preload.so PHANTOM_VERBOSE=1 \\"
echo "          ./llama.cpp/build/bin/llama-simple -m /path/to/model.gguf \\"
echo "          -n 16 -c 256 --no-warmup --no-repack"
echo ""
