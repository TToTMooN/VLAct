#!/bin/bash
# Policy server for the LIBERO-plus eval, 8xH200 box. Runs in the `vlact` env.
# Pair with eval_liberoplus_smoke.sh, which runs the simulator in `liberoplus`.
set -euo pipefail

VLACT_PYTHON=${VLACT_PYTHON:-/mnt/localssd/lingfeng/miniforge3/envs/vlact/bin/python}
CKPT=${CKPT:-playground/Pretrained_models/VLAct-LiberoPlus-PI/checkpoints/steps_50000_pytorch_model.pt}
PORT=${PORT:-9883}
GPU=${GPU:-0}

cd "$(dirname "$0")/../.."
# server_policy.py imports `deployment.*` as a package from the repo root.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=$GPU "$VLACT_PYTHON" deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT" \
    --port "$PORT" \
    --use_bf16
