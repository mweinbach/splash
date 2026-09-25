#!/bin/sh
# Builds hc.metallib (worker mk_hc + opt_qmv baselines) and hc_bench.
set -e
cd "$(dirname "$0")"
K=../../benchmarks/flash_opt_sep22/worker/kernels
F="-std=metal4.1 -O3 -mmacosx-version-min=27.0"
xcrun -sdk macosx metal $F -c $K/mk_hc.metal -o mk_hc.air
xcrun -sdk macosx metal $F -c $K/opt_qmv.metal -o opt_qmv.air
xcrun -sdk macosx metallib mk_hc.air opt_qmv.air -o hc.metallib
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation hc_bench.mm -o hc_bench
