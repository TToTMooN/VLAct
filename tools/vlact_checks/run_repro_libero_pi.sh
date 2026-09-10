#!/bin/bash
# Reproduce the LIBERO PI finetune from the VLAct continued-pretraining
# backbone on 8xH200. Defaults to a short smoke run; set STEPS=50000 for the
# full published recipe.
set -euo pipefail

cd "$(dirname "$0")/../.."

GPUS=${GPUS:-8}
STEPS=${STEPS:-200}
SAVE_EVERY=${SAVE_EVERY:-$STEPS}
BATCH=${BATCH:-16}
RUN_ID=${RUN_ID:-repro_libero_pi_${STEPS}steps}

export HF_HOME=${HF_HOME:-/mnt/localssd/lingfeng/cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-disabled}

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
