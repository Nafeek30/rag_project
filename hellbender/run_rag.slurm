#!/bin/bash
#SBATCH --job-name=rag-assistant
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/cluster/pixstor/home/mkfqm/rag_%j.log
#SBATCH --error=/cluster/pixstor/home/mkfqm/rag_%j.log

# =============================================================================
# RAG Knowledge Assistant — Hellbender SLURM Job
# - Qwen3 model cached in ~/ollama_models (downloaded once, reused every job)
# - Conda env + code built fresh in /tmp each job (no home quota pressure)
# Submit: sbatch hellbender/run_rag.slurm
# =============================================================================

set -e

WORK_DIR="/tmp/rag_${SLURM_JOB_ID}"
OLLAMA_BIN="$WORK_DIR/ollama"
OLLAMA_MODELS="$HOME/ollama_models"   # persistent — survives job restarts
CONDA_ENV="$WORK_DIR/conda_env"
API_PORT=8000
UI_PORT=8501
NODE=$(hostname -s)

echo "============================================================"
echo " RAG Assistant — Job $SLURM_JOB_ID on node $NODE"
echo " GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo " Time: $(date)"
echo "============================================================"

mkdir -p "$WORK_DIR" "$OLLAMA_MODELS" "$CONDA_ENV"

# ── 1. Clone project code ─────────────────────────────────────────
echo ""
echo "[1/5] Cloning RAG project to /tmp..."
git clone --depth=1 --branch feature/multi-model-claude-qwen3 --template=/dev/null \
    https://github.com/Nafeek30/rag_project.git "$WORK_DIR/rag_project"

# Load secrets from ~/.rag_secrets (never committed to git)
# Create it once on Hellbender: cp rag_project/.env.example ~/.rag_secrets && nano ~/.rag_secrets
SECRETS_FILE="$HOME/.rag_secrets"
if [ ! -f "$SECRETS_FILE" ]; then
    echo "ERROR: ~/.rag_secrets not found."
    echo "Create it with your API keys:"
    echo "  cp ~/rag_project/.env.example ~/.rag_secrets"
    echo "  nano ~/.rag_secrets"
    exit 1
fi
cp "$SECRETS_FILE" "$WORK_DIR/rag_project/.env"
echo "  Loaded secrets from ~/.rag_secrets"

# ── 2. Set up Python environment in /tmp ──────────────────────────
echo ""
echo "[2/5] Setting up Python environment in /tmp..."
module load miniconda3 2>/dev/null || true
conda create -p "$CONDA_ENV" python=3.12 -c conda-forge -y -q
source activate "$CONDA_ENV"
pip install -e "$WORK_DIR/rag_project" --quiet

# ── 3. Install Ollama to /tmp ─────────────────────────────────────
echo ""
echo "[3/5] Installing Ollama to /tmp..."
OLLAMA_VERSION="v0.22.0"
curl -fsSL "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" \
    -o "$WORK_DIR/ollama.tar.zst"
mkdir -p "$WORK_DIR/ollama_lib"
tar -I zstd -xf "$WORK_DIR/ollama.tar.zst" -C "$WORK_DIR"
mv "$WORK_DIR/bin/ollama" "$OLLAMA_BIN"
chmod +x "$OLLAMA_BIN"
mv "$WORK_DIR/lib/ollama" "$WORK_DIR/ollama_lib"
rm "$WORK_DIR/ollama.tar.zst"

# ── 4. Start Ollama — pull Qwen3 only if not cached ───────────────
echo ""
echo "[4/5] Starting Ollama..."
LD_LIBRARY_PATH="$WORK_DIR/ollama_lib:$LD_LIBRARY_PATH" \
OLLAMA_MODELS="$OLLAMA_MODELS" \
    "$OLLAMA_BIN" serve &>/tmp/ollama_${SLURM_JOB_ID}.log &
OLLAMA_PID=$!
sleep 5

if OLLAMA_MODELS="$OLLAMA_MODELS" "$OLLAMA_BIN" list 2>/dev/null | grep -q "qwen3:4b"; then
    echo "  Qwen3 4B already cached in ~/ollama_models — skipping download"
else
    echo "  Pulling Qwen3 4B (~2.5 GB) — this only happens once..."
    OLLAMA_MODELS="$OLLAMA_MODELS" "$OLLAMA_BIN" pull qwen3:4b
fi

# ── 5. Start FastAPI + Streamlit ──────────────────────────────────
echo ""
echo "[5/5] Starting services..."
cd "$WORK_DIR/rag_project"

OLLAMA_MODELS="$OLLAMA_MODELS" \
    uvicorn api.api:app --host 0.0.0.0 --port $API_PORT \
    &>/tmp/uvicorn_${SLURM_JOB_ID}.log &
UVICORN_PID=$!
sleep 5

streamlit run web_app/index.py \
    --server.port $UI_PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    &>/tmp/streamlit_${SLURM_JOB_ID}.log &
STREAMLIT_PID=$!
sleep 5

# ── Print tunnel instructions ─────────────────────────────────────
echo ""
echo "============================================================"
echo " ALL SERVICES RUNNING"
echo "============================================================"
echo ""
echo " Run this on YOUR LOCAL MACHINE to access the UI:"
echo ""
echo "   ssh -N -L ${UI_PORT}:${NODE}:${UI_PORT} -L ${API_PORT}:${NODE}:${API_PORT} mkfqm@hellbender.rnet.missouri.edu"
echo ""
echo " Then open: http://localhost:${UI_PORT}"
echo ""
echo " Logs:"
echo "   Ollama:    /tmp/ollama_${SLURM_JOB_ID}.log"
echo "   FastAPI:   /tmp/uvicorn_${SLURM_JOB_ID}.log"
echo "   Streamlit: /tmp/streamlit_${SLURM_JOB_ID}.log"
echo "   SLURM:     ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
echo "============================================================"

# ── Keep alive until cancelled ────────────────────────────────────
cleanup() {
    echo "Shutting down..."
    kill $STREAMLIT_PID $UVICORN_PID $OLLAMA_PID 2>/dev/null
    rm -rf "$WORK_DIR"
    exit 0
}
trap cleanup SIGTERM SIGINT

wait $STREAMLIT_PID $UVICORN_PID $OLLAMA_PID
