The private vocabulary dictionary preserves every BF16 coefficient used by the
current cached vocabulary projection, while retaining the original Q8 code
bytes. It replaces per-code F32 affine reconstruction and a BF16 cast with an
exact BF16 lookup; GPU reduction and complete-model performance still require
separate qualification.

The actual vocabulary shape is `[248320,2560]`, Q8 with group64. Its 9,932,800
groups contain only 41,151 distinct original BF16 `(scale,bias)` pairs, formed
from 952 scale values and 1,081 bias values. The pair dictionary therefore fits
uint16 group indices.

| Storage | Bytes |
| --- | ---: |
| Existing expanded BF16 vocabulary | 1,271,398,400 |
| Original Q8 code bytes, reused | 635,699,200 |
| Exact dictionary, BF16 `[41151,256]` | 21,069,312 |
| Group indices, U16 `[248320,40]` | 19,865,600 |
| Codes plus dictionary and indices | 676,634,112 |

The complete representation saves 594,764,288 bytes (46.78%) compared with the
expanded BF16 operand. Compared with the original packed Q8 codes plus
scale/bias bytes, it costs only 1,203,712 additional bytes. A private test that
keeps the existing BF16 cache alongside the candidate will temporarily add
40,934,912 bytes; it must disclose that allocation instead of claiming service
memory savings.

Both files are little endian and row major. Pair IDs follow ascending unsigned
`scale_bf16 | (bias_bf16 <<16)`. Decode is:

```text
pair = group_indices[n*40 + k/64]
code = original_code_bytes[n*2560 + k]
coefficient_bf16 = dictionary[pair*256 + code]
```

The CPU converter is `dev/tools/flash_vocab_bf16_dictionary.py`. It reads all
39,731,200 original scale/bias bytes and enumerates all 10,534,656 dictionary
coefficients. Explicit separate NumPy F32 multiply/add stages reproduce the
coefficient policy, followed by integer BF16 round to nearest, ties to even.
An independent BF16 encoder compares exact F64 affine values with adjacent
BF16 cells and parity at halfway values. For these actual source pairs, all
F32 affine sums equal their F64 results exactly, and all dictionary coefficients
match the independent encoder. A premature BF16 product cast would change
5,152,853 entries and is rejected by concrete source-pair traps.

The saved dictionary also matches 673,280 actual persisted BF16 coefficients
across 263 vocabulary rows, including EOS IDs and alignment boundaries. This
reads only 673,280 real code bytes and 1,346,560 saved operand bytes; it does
not scan the whole 635 MB code tensor or claim a fresh full-payload hash of the
1.27 GB saved operand. Source manifest hashes and file snapshots are verified;
the recorded original shard and cached-payload hashes remain provenance rather
than a new complete source-shard scan.

Private artifact: `build/flash-vocab-bf16-dictionary-v9/`.
CPU proof: `build/release/flash/vocab-bf16-dictionary-v9-cpu-qualification.json`.
No original checkpoint, original source package, production route, or GPU
command is changed by this converter.

```sh
.venv/bin/python -B dev/tools/flash_vocab_bf16_dictionary.py \
  --package install/local-models/Flash-Next-oQ4e-mtp-v1 \
  --output build/flash-vocab-bf16-dictionary-v9 \
  --operand-store install/local-models/Flash-Next-operands-v1 \
  --report build/release/flash/vocab-bf16-dictionary-v9-cpu-qualification.json

.venv/bin/python -B -m unittest \
  dev/tests/flash/test_flash_vocab_bf16_dictionary.py
```

Five focused CPU tests cover all 65,024 finite BF16 words, signed zero, both
signs at halfway boundaries and immediate neighbors, exact rational F32/BF16
affine reconstruction for all 256 codes, product-first convention traps,
nonfinite/overflow rejection, and uint16 signed-pair indexing.
