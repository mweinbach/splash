#!/bin/sh
# Builds pf2.metallib (mk_pf2 kernels + the worker's mk_prefill baseline) and pf2_bench.
set -e
cd "$(dirname "$0")"
W=../../benchmarks/flash_opt_sep22/worker
F="-std=metal4.1 -O3 -mmacosx-version-min=27.0 -I$W/source/runtime"
xcrun -sdk macosx metal $F -c mk_pf2.metal -o mk_pf2.air
xcrun -sdk macosx metal $F -c $W/kernels/mk_prefill.metal -o mk_prefill_worker.air
xcrun -sdk macosx metallib mk_pf2.air mk_prefill_worker.air -o pf2.metallib
clang++ -std=c++20 -O2 -fobjc-arc -I$W/source/runtime -framework Metal -framework Foundation pf2_bench.mm -o pf2_bench
