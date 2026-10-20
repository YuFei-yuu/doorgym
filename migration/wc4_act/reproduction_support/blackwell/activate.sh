# Source this file in Bash on ps1. All paths belong to this reproduction.
_video2door_restore_nounset=0
case $- in
    *u*) _video2door_restore_nounset=1; set +u ;;
esac
source /data/miniconda3/etc/profile.d/conda.sh
conda activate /data/zyf/video2door/.repro/envs/b1z1-blackwell
if [ "$_video2door_restore_nounset" = 1 ]; then set -u; fi
unset _video2door_restore_nounset
export CONDA_PKGS_DIRS=/data/zyf/video2door/.repro/conda-pkgs
export PIP_CACHE_DIR=/data/zyf/video2door/.repro/cache/pip
export XDG_CACHE_HOME=/data/zyf/video2door/.repro/blackwell/cache
export TORCH_EXTENSIONS_DIR=/data/zyf/video2door/.repro/blackwell/cache/torch_extensions
export CUDA_CACHE_PATH=/data/zyf/video2door/.repro/blackwell/cache/cuda
export TMPDIR=/data/zyf/video2door/.repro/tmp
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR"
export CUDA_HOME=/data/zyf/video2door/.repro/blackwell/cuda-12.8
export CUDA_PATH="$CUDA_HOME"
export NVTOOLSEXT_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export NVCC_PREPEND_FLAGS=""
export CUDNN_ROOT="$CONDA_PREFIX/lib/python3.8/site-packages/nvidia/cudnn"
export CUDNN_INCLUDE_DIR="$CUDNN_ROOT/include"
unset CUDNN_LIB_DIR
export CUDNN_LIBRARY="$CUDNN_ROOT/lib/libcudnn.so.9"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CUDNN_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VIDEO2DOOR_GPU=${VIDEO2DOOR_GPU:-0}
export CUDA_VISIBLE_DEVICES="$VIDEO2DOOR_GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export NODEVICE_SELECT=1
_video2door_gpu_uuid=$(nvidia-smi --id="$VIDEO2DOOR_GPU" --query-gpu=uuid --format=csv,noheader | sed 's/^GPU-//;s/-//g')
export VIDEO2DOOR_VK_DEVICE_INDEX=$(env -u VK_INSTANCE_LAYERS -u VK_LAYER_PATH /data/zyf/video2door/.repro/blackwell/probe_vulkan | awk -v uuid="$_video2door_gpu_uuid" '$NF == "UUID=" uuid {gsub(":", "", $2); print $2}')
unset _video2door_gpu_uuid
if [ -z "$VIDEO2DOOR_VK_DEVICE_INDEX" ]; then
    echo 'Unable to match the chosen CUDA GPU to a Vulkan GPU UUID.' >&2
    return 1
fi
export VK_LAYER_PATH=/data/zyf/video2door/.repro/blackwell/vulkan-layer
export VK_INSTANCE_LAYERS=VK_LAYER_VIDEO2DOOR_device_select
export VIDEO2DOOR_GRAPHICS_DEVICE_ID=0
export TORCH_CUDA_ARCH_LIST="12.0"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MAX_JOBS=16
cd /data/zyf/video2door

