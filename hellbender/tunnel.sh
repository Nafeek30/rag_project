#!/bin/bash
# =============================================================================
# Run this on YOUR LOCAL MACHINE to open the SSH tunnel to Hellbender
# Usage: bash hellbender/tunnel.sh <compute-node>
# Example: bash hellbender/tunnel.sh g004
#
# Get the node name from the SLURM job log:
#   squeue -u mkfqm               (shows running jobs)
#   cat rag-assistant_<jobid>.log  (shows node name at top)
# =============================================================================

NODE=${1:-""}

if [ -z "$NODE" ]; then
    echo "Usage: bash tunnel.sh <compute-node>"
    echo ""
    echo "Get the node name from your job log:"
    echo "  ssh mkfqm@hellbender.rnet.missouri.edu 'squeue -u mkfqm'"
    echo "  ssh mkfqm@hellbender.rnet.missouri.edu 'cat rag-assistant_*.log | head -5'"
    exit 1
fi

echo "============================================================"
echo " Opening SSH tunnel to Hellbender node: $NODE"
echo " Streamlit UI : http://localhost:8501"
echo " FastAPI docs : http://localhost:8000/docs"
echo " Press Ctrl+C to close the tunnel"
echo "============================================================"

ssh -N \
    -L 8501:${NODE}:8501 \
    -L 8000:${NODE}:8000 \
    mkfqm@hellbender.rnet.missouri.edu
