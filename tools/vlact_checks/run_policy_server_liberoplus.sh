#!/bin/bash
# Policy server for the LIBERO-plus eval. Defaults to GPU 4 so GPUs 0-3 stay
# free for other users on this shared box.
set -euo pipefail

VLACT_PYTHON=${VLACT_PYTHON:-/mnt/localssd/lingfeng/miniforge3/envs/vlact/bin/python}
CKPT=${CKPT:-playground/Pretrained_models/VLAct-LiberoPlus-PI/checkpoints/steps_50000_pytorch_model.pt}
PORT=${PORT:-9883}
GPU=${GPU:-4}

cd "$(dirname "$0")/../.."
# server_policy.py imports `deployment.*` as a package from the repo root.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=$GPU "$VLACT_PYTHON" deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT" \
    --port "$PORT" \
    --use_bf16
