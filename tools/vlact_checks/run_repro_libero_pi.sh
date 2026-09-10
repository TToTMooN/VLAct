#!/bin/bash
# Reproduce the LIBERO PI finetune from the VLAct continued-pretraining
# backbone. Defaults to GPUs 4-7 so GPUs 0-3 stay free for other users.
# Defaults to a short smoke run; set STEPS=50000 for the full published recipe.
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
GPUS=${GPUS:-4}
STEPS=${STEPS:-200}
SAVE_EVERY=${SAVE_EVERY:-$STEPS}
BATCH=${BATCH:-16}
RUN_ID=${RUN_ID:-repro_libero_pi_${STEPS}steps}

export HF_HOME=${HF_HOME:-/mnt/localssd/lingfeng/cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-disabled}
export PATH=/mnt/localssd/lingfeng/miniforge3/envs/vlact/bin:$PATH
# DeepSpeed probes CUDA_HOME/bin/nvcc on import even for ZeRO-2 with no custom ops.
export CUDA_HOME=/mnt/localssd/lingfeng/miniforge3/envs/vlact

OUT=results/Checkpoints/$RUN_ID
mkdir -p "$OUT"
cp "$0" "$OUT/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "$GPUS" \
  starVLA/training/train_starvla.py \
  --config_yaml ./tools/vlact_checks/repro_libero_pi.yaml \
  --run_id "$RUN_ID" \
  --datasets.vla_data.per_device_batch_size "$BATCH" \
  --trainer.max_train_steps "$STEPS" \
  --trainer.save_interval "$SAVE_EVERY" \
  --trainer.logging_frequency 10 \
  2>&1 | tee "$OUT/train.log"
