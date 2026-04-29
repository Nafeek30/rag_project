#!/bin/bash
#SBATCH --job-name=rag-assistant
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

# =============================================================================
# RAG Knowledge Assistant — Hellbender SLURM Job
# Submit: sbatch hellbender/run_rag.slurm
# =============================================================================

REPO_DIR="$HOME/rag_project"
OLLAMA_BIN="$HOME/bin/ollama"
OLLAMA_MODELS="$HOME/.ollama/models"
API_PORT=8000
UI_PORT=8501
NODE=$(hostname -s)

echo "============================================================"
echo " RAG Assistant starting on node: $NODE"
echo " Job ID: $SLURM_JOB_ID"
echo " GPU: $CUDA_VISIBLE_DEVICES"
echo " Time: $(date)"
echo "============================================================"

# ── Activate conda env ────────────────────────────────────────────
module load miniconda3 2>/dev/null || true
source activate rag-project || conda activate rag-project

# ── Start Ollama (local GPU inference) ───────────────────────────
echo ""
echo "[1/3] Starting Ollama on $NODE..."
OLLAMA_MODELS="$OLLAMA_MODELS" "$OLLAMA_BIN" serve &>/tmp/ollama_${SLURM_JOB_ID}.log &
OLLAMA_PID=$!
sleep 5

# Verify Ollama is up
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo "ERROR: Ollama failed to start. Check /tmp/ollama_${SLURM_JOB_ID}.log"
    exit 1
fi
echo "  Ollama running (PID $OLLAMA_PID)"

# ── Start FastAPI backend ─────────────────────────────────────────
echo ""
echo "[2/3] Starting FastAPI on port $API_PORT..."
cd "$REPO_DIR"
OLLAMA_MODELS="$OLLAMA_MODELS" \
    uvicorn api.api:app --host 0.0.0.0 --port $API_PORT \
    &>/tmp/uvicorn_${SLURM_JOB_ID}.log &
UVICORN_PID=$!
sleep 5

if ! curl -s http://localhost:$API_PORT/health &>/dev/null; then
    echo "ERROR: FastAPI failed to start. Check /tmp/uvicorn_${SLURM_JOB_ID}.log"
    exit 1
fi
echo "  FastAPI running (PID $UVICORN_PID)"

# ── Start Streamlit UI ────────────────────────────────────────────
echo ""
echo "[3/3] Starting Streamlit on port $UI_PORT..."
streamlit run "$REPO_DIR/web_app/index.py" \
    --server.port $UI_PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    &>/tmp/streamlit_${SLURM_JOB_ID}.log &
STREAMLIT_PID=$!
sleep 5
echo "  Streamlit running (PID $STREAMLIT_PID)"

# ── Print SSH tunnel instructions ─────────────────────────────────
echo ""
echo "============================================================"
echo " ALL SERVICES RUNNING"
echo "============================================================"
echo ""
echo " Run this on YOUR LOCAL MACHINE to open the tunnel:"
echo ""
echo "   ssh -N -L ${UI_PORT}:${NODE}:${UI_PORT} -L ${API_PORT}:${NODE}:${API_PORT} mkfqm@hellbender.rnet.missouri.edu"
echo ""
echo " Then open in your browser:"
echo "   http://localhost:${UI_PORT}"
echo ""
echo " Logs:"
echo "   Ollama:    /tmp/ollama_${SLURM_JOB_ID}.log"
echo "   FastAPI:   /tmp/uvicorn_${SLURM_JOB_ID}.log"
echo "   Streamlit: /tmp/streamlit_${SLURM_JOB_ID}.log"
echo "   SLURM:     ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
echo "============================================================"

# ── Keep job alive until cancelled ───────────────────────────────
cleanup() {
    echo "Shutting down services..."
    kill $STREAMLIT_PID $UVICORN_PID $OLLAMA_PID 2>/dev/null
    exit 0
}
trap cleanup SIGTERM SIGINT

wait $STREAMLIT_PID $UVICORN_PID $OLLAMA_PID
