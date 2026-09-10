#!/bin/bash
# Build the LIBERO-plus simulation environment from scratch, separate from the
# `vlact` env. The eval is two processes: the policy server runs in `vlact`
# (torch 2.6 + flash-attn), the simulator runs here (robosuite 1.4 + mujoco 2.3
# + numpy 1.24). One env cannot satisfy both, hence the split.
#
# The local SSD is ephemeral on this box, so this script has to be able to
# rebuild everything after a restart. Every version pin below is one that a
# failure taught us; see the comments.
set -euo pipefail

CONDA=${CONDA:-/mnt/localssd/lingfeng/miniforge3}
REPO=${REPO:-/mnt/localssd/lingfeng/LIBERO-plus}
ASSETS_ZIP=${ASSETS_ZIP:-/mnt/localssd/lingfeng/liberoplus_assets/assets.zip}
HF_PYTHON=${HF_PYTHON:-$CONDA/envs/vlact/bin/python}

echo "=== system deps (LIBERO-plus README + osmesa for headless mujoco) ==="
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev \
    libosmesa6-dev libgl1-mesa-glx libglew-dev patchelf unzip

echo "=== clone ==="
[ -d "$REPO" ] || git clone --depth 1 https://github.com/sylvestf/LIBERO-plus.git "$REPO"

echo "=== conda env liberoplus (python 3.10) ==="
"$CONDA/bin/conda" create -y -q -n liberoplus python=3.10
PY="$CONDA/envs/liberoplus/bin/python"
"$PY" -m pip install -q --upgrade pip

echo "=== LIBERO-plus requirements ==="
cd "$REPO"
"$PY" -m pip install -q -r requirements.txt
"$PY" -m pip install -q -e .
"$PY" -m pip install -q -r extra_requirements.txt

echo "=== VLAct client deps (examples/LIBERO-plus/README.md) ==="
"$PY" -m pip install -q tyro matplotlib mediapy websockets msgpack
# rich: starVLA's overwatch configures a RichHandler at import time and the
# client imports starVLA.model.tools for the unnormalization stats.
"$PY" -m pip install -q rich
# numpy: requirements.txt asks for 1.22.4, the VLAct client needs 1.24.4.
"$PY" -m pip install -q numpy==1.24.4
# mujoco: robosuite 1.4.0 only declares >=2.3.0, so pip takes 3.x, where
# robosuite's get_joint_qpos_addr() trips an assert on the joint type and every
# env build dies. 2.3.2 is the version robosuite 1.4 was written against.
"$PY" -m pip install -q 'mujoco==2.3.2'

echo "=== assets (hundreds of new objects/textures, ~9.5G unpacked) ==="
ASSET_DIR="$REPO/libero/libero/assets"
if [ ! -d "$ASSET_DIR" ]; then
    if [ ! -f "$ASSETS_ZIP" ]; then
        mkdir -p "$(dirname "$ASSETS_ZIP")"
        "$HF_PYTHON" -m huggingface_hub.commands.huggingface_cli download \
            Sylvest/LIBERO-plus --repo-type dataset --include 'assets.zip' \
            --local-dir "$(dirname "$ASSETS_ZIP")"
    fi
    # The zip carries the author's absolute path as its top-level directories,
    # so it cannot be unzipped straight into place.
    TMP=$(mktemp -d -p "$REPO")
    unzip -q -o "$ASSETS_ZIP" -d "$TMP"
    mv "$(find "$TMP" -type d -name assets -print -quit)" "$ASSET_DIR"
    rm -rf "$TMP"
fi

echo "=== libero config (LIBERO_CONFIG_PATH is a DIRECTORY) ==="
# Writing this up front keeps the first import from prompting interactively for
# a dataset path, which would hang any unattended run.
ROOT="$REPO/libero/libero"
mkdir -p "$REPO/datasets"
cat > "$REPO/libero/config.yaml" <<EOF
benchmark_root: $ROOT
bddl_files: $ROOT/bddl_files
init_states: $ROOT/init_files
datasets: $REPO/datasets
assets: $ROOT/assets
EOF

echo "=== verify ==="
cd "$REPO"
LIBERO_CONFIG_PATH="$REPO/libero" MUJOCO_GL=osmesa "$PY" - <<'EOF'
import mujoco, numpy, robosuite
from libero.libero import benchmark
print("numpy", numpy.__version__, "| robosuite", robosuite.__version__, "| mujoco", mujoco.__version__)
suite = benchmark.get_benchmark_dict()["libero_goal"]()
print("libero_goal tasks:", suite.n_tasks, "| task 0:", suite.get_task(0).language)
EOF
echo "LIBEROPLUS_SETUP_OK"
