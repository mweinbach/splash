#!/bin/bash
# name K N bits group
LIB=${LIB:-opt.metallib}
FILTER=${FILTER:-opt_qmv}
while read name K N bits G; do
  for r in ${ROWS:-1 5}; do
    printf "%-14s r=%d " $name $r
    ./qmv_bench $LIB $K $N $bits $G $r $FILTER 2>/dev/null | sort -k2 -n | head -1
  done
done <<'SHAPES'
HCdown-q4 10240 320 4 64
HCdown-q5 10240 320 5 64
qkv-q6 2560 10240 6 64
qkv-q5 2560 10240 5 64
z-q5 2560 6144 5 128
out-q5 6144 2560 5 128
qproj-q6 2560 12288 6 64
oproj-q6 6144 2560 6 64
shexp-gu-q8 2560 640 8 128
shexp-dn-q8 640 2560 8 128
ab-q5 2560 48 5 128
HCup-q5 320 10240 5 64
SHAPES
