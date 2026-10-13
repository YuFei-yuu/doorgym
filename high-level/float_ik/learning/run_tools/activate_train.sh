# Source in a fresh Bash child; never changes the shared Conda base environment.
source /data/miniconda3/etc/profile.d/conda.sh || return
conda activate /data/zyf/video2door/.repro/envs/door-dp-train || return
export CONDA_PKGS_DIRS=/data/zyf/video2door/.repro/conda-pkgs
export PIP_CACHE_DIR=/data/zyf/video2door/.repro/cache/pip
export XDG_CACHE_HOME=/data/zyf/video2door/.repro/cache/train
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES=${VIDEO2DOOR_GPU:?Select a physical GPU through launch.py}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
unset CUDA_HOME CUDA_PATH CUDACXX CUDNN_ROOT CUDNN_LIBRARY VK_INSTANCE_LAYERS VK_LAYER_PATH
cd /data/zyf/video2door
