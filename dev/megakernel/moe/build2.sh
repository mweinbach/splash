#!/bin/sh
# Builds moe2.metallib (mk2 kernels + the worker's mk_moe baseline) and moe2_bench.
set -e
cd "$(dirname "$0")"
F="-std=metal4.1 -O3 -mmacosx-version-min=27.0"
xcrun -sdk macosx metal $F -c mk_moe2.metal -o mk_moe2.air
xcrun -sdk macosx metal $F -c ../../benchmarks/flash_opt_sep22/worker/kernels/mk_moe.metal -o mk_moe_worker.air
xcrun -sdk macosx metallib mk_moe2.air mk_moe_worker.air -o moe2.metallib
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation moe2_bench.mm -o moe2_bench
