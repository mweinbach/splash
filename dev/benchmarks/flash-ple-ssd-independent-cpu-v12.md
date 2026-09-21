# Optional SSD PLE: independent CPU qualification

These checks exercise the production host hash, native-window layout planner,
and disk row store. They do not create a Metal device, map a PLE table, load a
model, or qualify GPU/model/HTTP behavior. All original files remain unchanged.

The source contract is the local oMLX implementation at
`/Users/mweinbach/Projects/omlx/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py`:
`Qwen4ExpNGramEmbedding._shift_right_ignore_eos`, `_ngram_indices`, and
`DiskBackedShardedEmbedding._call_owned`. The independent expected hashes use
an explicit segmented shifted-token construction; the production host hash
uses a two-token recurrence. Both wrap products as I64 and use nonnegative
signed remainder. Current EOS retains prior context; the following token
resets the segment. PLE EOS is 248044. Generation stop 248046 does not reset
PLE context.

Run the production hash/layout ASan and UBSan checks:

```sh
.venv/bin/python -B dev/tests/flash/run_flash_ple_ssd_independent_cpu.py \
  --sanitize --build build/flash-ple-ssd-independent-cpu-v12-sanitized
```

The report at
`build/flash-ple-ssd-independent-cpu-v12-sanitized/qualification.json` records
1,633,484 checks against 1,613,072 independently generated IDs, using just
282 checkpoint bytes of stored I64/BF16 metadata. Cases include B1 through B4,
real R1/2/4/16/512/2048, ragged histories, chunk transitions, each retained
prefix 0 through R followed by future tokens, consecutive EOS and distinct
generation stops, signed overflowing products and INT64_MIN, invalid extents
and tokens, and history immutability on rejection. Prefix cases prove hashing
from an already restored history; they do not execute a state restoration.

The actual manifest independently partitions into 28 native windows totaling
74,317,889,536 bytes and 384 disk-only planes totaling 32,002,539,520 padded
bytes. Every native window is disjoint from every PLE plane's complete 16 KiB
page extent. Every non-PLE tensor belongs to exactly one native window. Disk
logical bytes are 32,000,153,600. The report includes all per-window tensor
owners. Static membership does not prove Metal view/mapping lifetime retention;
the Root loader oracle covers actual native allocations and owners.

Run the independent 200-row synthetic planar store proof:

```sh
xcrun clang++ -std=c++20 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -Iruntime \
  dev/tests/flash/flash_ple_ssd_store_independent_cpu.cpp \
  runtime/flash/FlashPLESSDStore.cpp \
  -o build/flash-ple-ssd-independent-cpu-v12-sanitized/store-independent-cpu-oracle
build/flash-ple-ssd-independent-cpu-v12-sanitized/store-independent-cpu-oracle \
  build/flash-ple-ssd-independent-cpu-v12-sanitized/store-fixture \
  > build/flash-ple-ssd-independent-cpu-v12-sanitized/store-qualification.json
```

This passes 45,697,632 checks and compares 46,653,900 bytes. It covers 4,800
randomized batches plus 400 concurrent shared-store batches, all shard
boundaries, original input order and duplicates, zero-cache mode, cache hits,
LRU churn and cache clearing. Cache budgets are 0/1 KiB/4 KiB/1 MiB; read
scratch limits are 80/128/1024 bytes. A measured nine-ID duplicate case reads
exactly four unique 100-byte rows in six adjacent plane reads. Invalid IDs
leave output unchanged, do not read sources, and do not poison the healthy
store. Source replacement/truncation/identity and constructor-overflow
adversaries are independently covered by the store owner's contract audit.

Run the bounded actual checkpoint store and reconstruction proof when the
Root permits sparse CPU reads alongside its serial GPU/cache measurements:

```sh
.venv/bin/python -B dev/tests/flash/run_flash_ple_ssd_checkpoint_store_cpu.py
```

`build/flash-ple-ssd-independent-checkpoint-v12/qualification.json` records
the first and last row of each of the original 128 parts: 256 rows, 25,600
packed bytes, and 40,960 exactly equal BF16 outputs. The production reader
uses 768 reads for 25,600 bytes; the independent reference reads another
25,600 bytes and the shared BF16 scale reads two bytes. A warm repeat obtains
256 cache hits without another source read. The configured 64 MiB cache
accounts 50,331,648 bytes in fixed arrays and retains 256 used rows. Decoding
compares independent planar U32 nibble extraction against byte staging,
preserving F32 `q*scale+bias`, BF16 reconstruction, shared BF16 scale, then
BF16 output. No full-table hashing or interpretation of these reads as
physical SSD misses is claimed. `F_NOCACHE` is an OS file-cache policy.

Root still needs to execute the actual Metal primitive and complete model
singleton/batch prefill, verification/partial restore, cancellation/deadline,
quality and HTTP throughput checks before declaring the optional route
qualified.
