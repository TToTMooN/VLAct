#!/bin/bash
# Build the LIBERO-plus simulation environment, separate from the `vlact` env.
# The eval is two processes: the policy server runs in `vlact` (torch 2.6 +
# flash-attn), the simulator runs here (mujoco/robosuite with pinned old deps).
# Mixing them in one env is what breaks; hence a second conda env.
set -euo pipefail

CONDA=/mnt/localssd/lingfeng/miniforge3
REPO=/mnt/localssd/lingfeng/LIBERO-plus

echo "=== system deps (LIBERO-plus README + osmesa for headless mujoco) ==="
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev \
    libosmesa6-dev libgl1-mesa-glx libglew-dev patchelf

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
# Pinned last: requirements.txt asks for 1.22.4, the VLAct client needs 1.24.4.
"$PY" -m pip install -q numpy==1.24.4

echo "=== verify ==="
cd "$REPO"
MUJOCO_GL=osmesa "$PY" - <<'EOF'
import numpy, robosuite, mujoco
import libero.libero as ll
from libero.libero import benchmark
print("numpy", numpy.__version__, "| robosuite", robosuite.__version__, "| mujoco", mujoco.__version__)
suites = benchmark.get_benchmark_dict()
print("benchmark suites:", len(suites))
print("sample:", sorted(suites)[:8])
EOF
echo "LIBEROPLUS_SETUP_OK"
