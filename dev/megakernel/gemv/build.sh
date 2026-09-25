#!/bin/sh
# Builds gemv.metallib (new kernels + the worker's opt_qmv/opt_mppq baselines) and gemv_bench.
set -e
cd "$(dirname "$0")"
K=../../benchmarks/flash_opt_sep22/worker/kernels
F="-std=metal4.1 -O3 -mmacosx-version-min=27.0"
xcrun -sdk macosx metal $F -c mk_qmv.metal -o mk_qmv.air
xcrun -sdk macosx metal $F -c mk_mpt.metal -o mk_mpt.air
xcrun -sdk macosx metal $F -c $K/opt_qmv.metal -o opt_qmv.air
xcrun -sdk macosx metal $F -c $K/opt_mppq.metal -o opt_mppq.air
xcrun -sdk macosx metallib mk_qmv.air mk_mpt.air opt_qmv.air opt_mppq.air -o gemv.metallib
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation gemv_bench.mm -o gemv_bench
