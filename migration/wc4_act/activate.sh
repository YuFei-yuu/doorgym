#!/usr/bin/env bash
# Source in a fresh Bash child. Old environments and support remain in place.
_wc4_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
_wc4_kind=${1:?Usage: source migration/wc4_act/activate.sh gym|act}
export PYTHONNOUSERSITE=1
unset PYTHONPATH
if [[ "$_wc4_kind" == gym ]]; then
    : "${VIDEO2DOOR_GPU:?Select an idle physical GPU before activation}"
    source /data/zyf/video2door/.repro/blackwell/activate.sh || return
elif [[ "$_wc4_kind" == act ]]; then
    source /data/miniconda3/etc/profile.d/conda.sh || return
    conda activate "$_wc4_root/.repro/envs/door-act" || return
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
    unset CUDA_HOME CUDA_PATH CUDACXX CUDNN_ROOT CUDNN_LIBRARY VK_INSTANCE_LAYERS VK_LAYER_PATH
    export CUDA_VISIBLE_DEVICES=${VIDEO2DOOR_GPU:-}
else
    echo 'Expected gym or act' >&2
    return 2
fi
export DOOR_DP_LEROBOT_PYTHON="$_wc4_root/.repro/envs/door-act/bin/python"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PIP_CACHE_DIR="$_wc4_root/.repro/cache/pip"
export XDG_CACHE_HOME="$_wc4_root/.repro/cache/$_wc4_kind"
export TMPDIR="$_wc4_root/.repro/tmp"
export WANDB_MODE=online WANDB_ENTITY=fi-yu-individual WANDB_PROJECT=door-dp-stop-tail
export WANDB_RUN_GROUP=wc4-act-joint9-skill-baseline-20261015
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE WANDB_DISABLED WANDB_RUN_ID WANDB_RESUME
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"
cd "$_wc4_root" || return
unset _wc4_root _wc4_kind
