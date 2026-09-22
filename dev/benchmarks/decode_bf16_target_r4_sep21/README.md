The private worker builds over the exact pointwise + gathered + bulk Full512
source snapshot. It selects only the four measured source tuples at physical
rows4. Other row counts, roles and quantizations preserve the original route.

| Main role | Output / input | Source format | New BF16 tile |
| --- | --- | --- | --- |
| GDN QKV | 10240 / 2560 | Q6 G64 | M8 N64 SG4 |
| GDN Z | 6144 / 2560 | Q6 G64 | M8 N32 SG2 |
| PLE value | 2560 / 2560 | Q4 G64 | M8 N32 SG2 |
| QSA Q | 12288 / 2560 | Q4 G64 | M8 N32 SG2 |

Source codes must be U32, source scales/biases BF16 and experts1. Cached weights
must be the original derived BF16 matrix with exact admitted dimensions. The
existing BF16 small-row host API preserves view/ownership/alias guards before
encoding the original positive-zero pad and the selected whole-K producer.
The standalone kernel source is unchanged from Root's exact cached-BF16/F32-dot
screen. Cached BF16 precision remains a numerical alternative to the raw/F32
body coefficients, requiring complete model/MTP acceptance checks.

```sh
.venv/bin/python dev/benchmarks/decode_bf16_target_r4_sep21/overlay.py
make -f dev/benchmarks/decode_bf16_target_r4_sep21/Makefile -j4 cpu
.venv/bin/python dev/benchmarks/decode_bf16_target_r4_sep21/witness.py \
  build/bf16-target-r4-pointwise-sep21-v2
```

Root launches with current pointwise/gathered/exact-bulk profile and strict
`SPLASH_FLASH_DECODE_BF16_R4_TARGETED=0/1`. Default0 preserves the old numerical
identity seed. Enabled requires DENSE_CACHE1, FLOAT_DENSE_CACHE1 and
DENSE_SMALL_ROWS0. Legacy `SPLASH_FLASH_DECODE_BF16_DENSE=1` is forbidden.
Flag/scope are fixed per process; enabled adds one independently admitted1MiB
padding workspace and binds the exact tuple/tile/precision policy into the
target derivative. HC, trained MTP, vocabulary and full2048 prefill remain
unchanged. Eligible4-row prefill suffixes and batch calls use the same policy.

Static enabled/policy status remains in immutable identity. Mutable graph-call/
row counts are in top-level `target_bf16_r4_route_counters`; they describe graph
construction, not completed GPU work. This source audit rejects mutable
counters inside identity. Source, header, reused object/core/AIR closure is
frozen and hashed; Forward, Store and Worker implementations rebuild, plus the
three batch implementations consuming the additive Forward header getters.
CPU preparation touches no GPU or model payload. Root owns full generation,
state, lifecycle, prefill/decode and acceptance qualification.
