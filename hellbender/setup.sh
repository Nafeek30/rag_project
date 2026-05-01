#!/bin/bash
# =============================================================================
# Hellbender ONE-TIME setup — run this on the LOGIN NODE once
# Usage: bash hellbender/setup.sh
# =============================================================================
set -e

REPO_DIR="$HOME/rag_project"
OLLAMA_BIN="$HOME/bin/ollama"
OLLAMA_MODELS="$HOME/.ollama/models"

echo "============================================================"
echo " RAG Project — Hellbender Setup"
echo "============================================================"

# ── 1. Clone the repo ────────────────────────────────────────────
echo ""
echo "[1/5] Cloning repository..."
if [ -d "$REPO_DIR" ]; then
    echo "  Repo already exists — pulling latest..."
    git -C "$REPO_DIR" pull
else
    git clone https://github.com/Nafeek30/rag_project.git "$REPO_DIR"
    cd "$REPO_DIR" && git checkout feature/multi-model-claude-qwen3
fi
cd "$REPO_DIR"

# ── 2. Create conda environment ───────────────────────────────────
echo ""
echo "[2/5] Setting up conda environment (Python 3.12)..."
module load miniconda3 2>/dev/null || true
conda create -n rag-project python=3.12 -c conda-forge -y || echo "  Env already exists, skipping."
source activate rag-project || conda activate rag-project

echo "  Installing Python dependencies..."
pip install -e "$REPO_DIR" --quiet

# ── 3. Create .env file ───────────────────────────────────────────
echo ""
echo "[3/5] Creating .env file..."
ENV_FILE="$REPO_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" <<'EOF'
LLM_PROVIDER=groq

GROQ_API_KEY=
ANTHROPIC_API_KEY=
GROK_API_KEY=

PINECONE_API_KEY=pcsk_2EXjHN_ErxJtWhCSKRcdDnyacuAxRyvuqfLsr88pGpLVytALkHYYei3ZioCP6nkqo9fWZj
PINECONE_INDEX_NAME=rag-knowledge

OLLAMA_HOST=http://localhost:11434
EOF
    echo "  .env created — add your GROQ/ANTHROPIC keys if needed."
else
    echo "  .env already exists, skipping."
fi

# ── 4. Install Ollama ─────────────────────────────────────────────
echo ""
echo "[4/5] Installing Ollama..."
mkdir -p "$HOME/bin" "$HOME/lib/ollama"

OLLAMA_VERSION="v0.22.0"
TARBALL="/tmp/ollama.tar.zst"

curl -fsSL "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" \
    -o "$TARBALL"

mkdir -p /tmp/ollama_extract
tar -I zstd -xf "$TARBALL" -C /tmp/ollama_extract
cp /tmp/ollama_extract/bin/ollama "$OLLAMA_BIN"
chmod +x "$OLLAMA_BIN"
rsync -a /tmp/ollama_extract/lib/ollama/ "$HOME/lib/ollama/"
rm -rf /tmp/ollama_extract "$TARBALL"

echo "  Ollama installed at $OLLAMA_BIN"

# ── 5. Pull Qwen3 4B ──────────────────────────────────────────────
echo ""
echo "[5/5] Pulling Qwen3 4B model (~2.5 GB)..."
echo "  Starting temporary Ollama server..."
OLLAMA_MODELS="$OLLAMA_MODELS" "$OLLAMA_BIN" serve &>/tmp/ollama_setup.log &
OLLAMA_PID=$!
sleep 5

OLLAMA_MODELS="$OLLAMA_MODELS" "$OLLAMA_BIN" pull qwen3:4b
kill $OLLAMA_PID 2>/dev/null

echo ""
echo "============================================================"
echo " Setup complete!"
echo " Next step: submit the job with:"
echo "   sbatch $REPO_DIR/hellbender/run_rag.slurm"
echo "============================================================"
