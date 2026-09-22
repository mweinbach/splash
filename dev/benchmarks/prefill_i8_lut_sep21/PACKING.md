`pack.py` creates one bounded layer containing all 512 experts and all three
projections. It preserves the certified Full512 signed-I8 code bytes and the
original late F32 row scales. It does not requantize weights, construct a model
overlay, or run GPU work.

From the Splash checkout, use Python 3.12+ with NumPy:

```sh
.venv/bin/python dev/benchmarks/prefill_i8_lut_sep21/pack.py --cpu-self-test
.venv/bin/python dev/benchmarks/prefill_i8_lut_sep21/pack.py inspect \
  --source install/local-models/Flash-Next-oQ4e-mtp-v1 \
  --store build/prefill4k-fullcache-artifacts/int8-experts-all512-v1 \
  --layer 0
```

The self-test uses small synthetic arrays/files only. `inspect` opens bounded
JSON metadata and calls `stat`; it does not open or hash model payloads.

Only the root experiment runner should execute payload packing:

```sh
.venv/bin/python dev/benchmarks/prefill_i8_lut_sep21/pack.py pack \
  --source install/local-models/Flash-Next-oQ4e-mtp-v1 \
  --store build/prefill4k-fullcache-artifacts/int8-experts-all512-v1 \
  --layer 0 \
  --output build/prefill-i8-lut-sep21-layer00-v1 \
  --batch-experts 4
```

The output parent must exist and the output directory must be fresh and
separate from the source/store. The packer stages `layer-00.bin`,
`manifest.json`, and `certificate.json` in a sibling temporary directory,
checks the finished file, flushes all files, and atomically publishes the
directory with an exclusive rename that cannot replace a concurrent empty
destination. An unsuccessful pre-publication run removes its temporary output.
The CLI accepts exactly one `--layer` in `[0,47]` and batches `[1,8]` experts.
It has no full-model packing option. Certificate and conversion report paths
default to the existing store's `coefficient-certificate.json` and
`conversion-report.json`; explicit paths use `--certificate` and
`--conversion-report`.

The metadata schema is `splash-prefill-i8-lut-one-layer-v1`. Rank is the
original expert ID in complete order `0..511`. In gate/up/down order, every
projection has three separately 16-KiB-aligned planes:

| Plane | Dtype and shape | Order | Bytes per G64 |
| --- | --- | --- | ---: |
| `ids` | U8 `[512,N,K/2]` | rank, output row, packed input byte | 32 |
| `lut` | I8 `[512,N,K/64,16]` | rank, output row, input group, Q4 ID | 16 |
| `scales` | F32 `[512,N]` | rank, output row | Original 4 bytes per row |

IDs are the original little-endian U32 weight bytes, with even input
coordinates in the low nibble and odd coordinates in the high nibble.
Within each G64 group, the LUT maps each original Q4 ID to its saved signed-I8
code. Multiple Q4 IDs may map to one I8 code after quantization. A Q4 ID that
appears more than once must always map to the same code in that group;
otherwise packing fails. LUT entries for unobserved IDs are canonical zero.
The packer never introduces code `-128`.

Each plane descriptor has `dtype`, `shape`, `offset`, `length`, and `sha256`.
The manifest's `path`, `bytes`, and `sha256` describe the packed binary.
`source_i8_layer` carries the absolute path, `logical_size`, complete binary
hash, and original code/scale descriptors for the chosen existing I8 layer.
`original_q4_id_inputs` retains the original tensor extents, certified shard
identities, certified file snapshots, and hashes of the chosen ID extents.

The weight payload is exactly 48 rather than 64 bytes per G64 group: a 25%
reduction. For one complete Flash layer, code payload falls from
2,516,582,400 to 1,887,436,800 bytes. The original 7,864,320 scale bytes remain,
so the packed binary is 1,895,301,120 bytes versus 2,524,446,720 bytes for the
existing I8 binary. A full 90-GB sidecar is not created.

Source identity, both manifest hashes, the coefficient-certificate hash, and
the conversion-report hash are pinned to the existing published source and
Full512 store. The packer requires the complete published coefficient
certificate, its finite coefficient error metrics and bound, and the
conversion-report full-shard verification. Source
shard device/inode/size/mtime/ctime must match the old certified snapshots
before and after packing. The original source shard permissions are preserved
(the installed source uses `0644`), and access uses `O_RDONLY|O_NOFOLLOW`.
The saved I8 layer and emitted binary must be read-only regular files.

Only the chosen layer's original Q4 ID extents and saved I8 layer are read.
The whole chosen I8 binary and every original I8 code/scale plane are checked
against the published hashes. Source BF16 scale/bias tensors are validated by
metadata and certified shard snapshots; their payloads are not read. No
source shard or full-store rehash is performed.

Every saved code byte is checked twice with an independent LE-U32 word/shift
decoder: once before output writes and again by reading the finished output.
The byte-nibble algorithm that derives the LUT is separate from this decoder.
Finished output IDs/LUT/scales have independently verified plane hashes, and
every copied scale byte is compared to the existing scale plane. The emitted
certificate schema is `splash-prefill-i8-lut-exact-byte-certificate-v1` and
binds the raw JSON manifest hash, all three full projection cardinalities,
complete code reconstruction, complete scale copying, disk readback, and
unchanged input snapshots. This certifies the representation; GPU arithmetic
and speed still require the component oracle.
