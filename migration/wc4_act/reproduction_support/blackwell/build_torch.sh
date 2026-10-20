#!/usr/bin/env bash
set -euo pipefail
source /data/zyf/video2door/.repro/blackwell/activate.sh
export USE_CUDA=1
export USE_CUDNN=1
export USE_DISTRIBUTED=0
export USE_NCCL=0
export USE_GLOO=0
export USE_MPI=0
export USE_TENSORPIPE=0
export USE_KINETO=0
export USE_MKL=0
export USE_MKLDNN=0
export USE_FBGEMM=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0
export USE_XNNPACK=0
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0
export USE_CUSPARSELT=0
export BUILD_TEST=0
export BUILD_CAFFE2=0
export BUILD_CAFFE2_OPS=0
export BUILD_BINARY=0
export USE_CUDA_STATIC_LINK=0
export BLAS=OpenBLAS
export CMAKE_BUILD_TYPE=Release
export _GLIBCXX_USE_CXX11_ABI=0
export PYTORCH_BUILD_VERSION=2.3.1+sm120
export PYTORCH_BUILD_NUMBER=1
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export NVCC_PREPEND_FLAGS=""
cd /data/zyf/video2door/.repro/blackwell/src/pytorch
python /data/zyf/video2door/.repro/blackwell/patch_pytorch.py
git diff --ignore-submodules > /data/zyf/video2door/.repro/blackwell/pytorch-sm120.patch
nvcc --version
nice -n 10 python setup.py bdist_wheel

