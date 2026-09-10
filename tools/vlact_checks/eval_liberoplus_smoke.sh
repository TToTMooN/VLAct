#!/bin/bash
# Tiny LIBERO-plus slice, NOT the 10,030-task benchmark: a few tasks x a few
# trials, enough to prove the server/simulator round trip and unnormalization.
# Runs in the `liberoplus` env; start run_policy_server_liberoplus.sh first.
set -euo pipefail

LIBERO_PYTHON=${LIBERO_PYTHON:-/mnt/localssd/lingfeng/miniforge3/envs/liberoplus/bin/python}
export LIBERO_HOME=${LIBERO_HOME:-/mnt/localssd/lingfeng/LIBERO-plus}
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export MUJOCO_GL=osmesa

# Must be the .pt file: read_mode_config() takes parents[1] of it to find
# config.yaml and dataset_statistics.json for unnormalization.
CKPT=${CKPT:-playground/Pretrained_models/VLAct-LiberoPlus-PI/checkpoints/steps_50000_pytorch_model.pt}
PORT=${PORT:-9883}
SUITE=${SUITE:-libero_goal}
TASK_START=${TASK_START:-0}
TASK_END=${TASK_END:-2}
TRIALS=${TRIALS:-2}

cd "$(dirname "$0")/../.."
# The client imports the websocket helpers from the repo root and the benchmark
# from the LIBERO-plus checkout.
export PYTHONPATH="$(pwd):${LIBERO_HOME}:${PYTHONPATH:-}"

OUT=results/liberoplus_smoke/$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

"$LIBERO_PYTHON" ./examples/LIBERO-plus/eval_files/eval_libero.py \
    --args.pretrained-path "$CKPT" \
    --args.host 127.0.0.1 \
    --args.port "$PORT" \
    --args.task-suite-name "$SUITE" \
    --args.task-id-start "$TASK_START" \
    --args.task-id-end "$TASK_END" \
    --args.num-trials-per-task "$TRIALS" \
    --args.video-out-path "$OUT/videos" \
    --args.log-path "$OUT" \
    2>&1 | tee "$OUT/eval.log"
