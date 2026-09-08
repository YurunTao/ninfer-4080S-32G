#!/bin/bash
# Build NInfer-4090 (sm_89) for RTX 4080S.
#
# ROOT CAUSE of earlier failures: /usr/bin/cudafe++ (from the legacy
# nvidia-cuda-toolkit pkg, CUDA 12.4 generation) shadowed
# /usr/local/cuda-13.0/bin/cudafe++ in PATH. nvcc 13.0 passes
# --static-host-stub to cudafe++; the old cudafe++ rejects it with
# "Command-line error: invalid option: --static-host-stub".
# FIX: put /usr/local/cuda-13.0/bin FIRST in PATH.
#
# Host compiler: gcc-13 (nvcc 13.0 rejects gcc>13; system default is gcc-15).
set -uo pipefail

export PATH=/usr/local/cuda-13.0/bin:/usr/bin:/bin:/usr/sbin:/sbin
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc
export CUDAHOSTCXX=/usr/bin/g++-13

cd /home/raymond/ninfer-4090
rm -rf build-sm89

cmake -S . -B build-sm89 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DNINFER_BUILD_APPS=ON \
  -DBUILD_TESTING=OFF \
  -DNINFER_BUILD_BENCHMARKS=OFF
CFG=$?
echo "=== CMAKE CONFIGURE exit=$CFG ==="
if [ $CFG -ne 0 ]; then exit $CFG; fi

cmake --build build-sm89 --parallel 4
echo "=== BUILD exit=$? ==="
ls -la build-sm89/apps/ 2>&1
