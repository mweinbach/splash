#!/bin/bash
# Full variant sweep: prints every variant timing per shape/rows.
while read name K N bits G; do
  for r in 1 2 3 4 5 8; do
    ./qmv_bench opt_qmv.metallib $K $N $bits $G $r opt_qmv 2>/dev/null | sed "s/^/$name $K $N $bits $G r$r /"
  done
done <<'SHAPES'
HCdown-q4 10240 320 4 64
HCdown-q6 10240 320 6 64
qkv-q6 2560 10240 6 64
qkv-q5 2560 10240 5 64
z-q5 2560 6144 5 128
z-q6 2560 6144 6 64
out-q5 6144 2560 5 128
qproj-q6 2560 12288 6 64
qproj-q8 2560 12288 8 64
oproj-q6 6144 2560 6 64
kv-q8 2560 512 8 64
idx-q6 2560 640 6 64
shexp-gu-q8 2560 640 8 128
shexp-dn-q8 640 2560 8 128
ab-q5 2560 48 5 128
HCup-q5 320 10240 5 64
fc-q4 5120 2560 4 64
default-q4 2560 2560 4 64
SHAPES
