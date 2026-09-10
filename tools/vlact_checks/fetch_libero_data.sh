#!/bin/bash
# Fetch the four LIBERO LeRobot datasets, working around HF's request quota.
#
# These datasets are thousands of small files (one video per episode), so a
# single download burns through the authenticated limit of 1000 API requests
# per 5 minutes long before it finishes. hf download resumes, so retrying with
# a cooldown converges; --max-workers keeps the request rate down.
#
# Completion is decided by comparing meta/info.json's counts against the files
# on disk, NOT by the exit code: a rate-limited download was observed exiting 0
# having fetched no meta/ and no videos at all.
set -uo pipefail

export HF_HOME=${HF_HOME:-/mnt/localssd/lingfeng/cache/huggingface}
DEST=${DEST:-/mnt/localssd/lingfeng/datasets}/libero
HF=${HF:-/mnt/localssd/lingfeng/miniforge3/envs/vlact/bin/hf}
PY=${PY:-/mnt/localssd/lingfeng/miniforge3/envs/vlact/bin/python}
COOLDOWN=${COOLDOWN:-320}   # just over HF's 5-minute window
MAX_ATTEMPTS=${MAX_ATTEMPTS:-40}

REPOS=(
  IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
  IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
)

is_complete() {
    "$PY" - "$1" <<'EOF'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
try:
    info = json.loads((root / "meta" / "info.json").read_text())
except (OSError, ValueError):
    print("no meta/info.json"); sys.exit(1)
videos = len(list(root.rglob("*.mp4")))
parquets = len(list(root.rglob("*.parquet")))
want_v, want_e = info.get("total_videos", 0), info.get("total_episodes", 0)
if videos < want_v or parquets < want_e:
    print(f"videos {videos}/{want_v}, parquet {parquets}/{want_e}"); sys.exit(1)
print(f"{want_e} episodes, {want_v} videos")
EOF
}

mkdir -p "$DEST"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    remaining=0
    for repo in "${REPOS[@]}"; do
        name=${repo##*/}
        dir="$DEST/$name"
        if status=$(is_complete "$dir" 2>/dev/null); then
            continue
        fi
        echo "[$attempt] $name: $status"
        $HF download "$repo" --repo-type dataset \
            --local-dir "$dir" --max-workers 4 >> "/tmp/dl_$name.log" 2>&1
        if status=$(is_complete "$dir" 2>/dev/null); then
            echo "[$attempt] $name COMPLETE — $status ($(du -sh "$dir" | cut -f1))"
        else
            remaining=$((remaining + 1))
            echo "[$attempt] $name still incomplete — $status"
        fi
    done
    if [ "$remaining" -eq 0 ]; then
        echo "ALL_COMPLETE"
        du -sh "$DEST"/*
        exit 0
    fi
    echo "[$attempt] $remaining left, sleeping ${COOLDOWN}s for the quota window"
    sleep "$COOLDOWN"
done
echo "GAVE_UP after $MAX_ATTEMPTS attempts"
exit 1
