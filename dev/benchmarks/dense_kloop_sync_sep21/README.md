Private BF16 dense prefill synchronization screen. Production files remain
unchanged. Only Root invokes `--run`; compilation, preview and CPU self-test
never create a GPU backend or read captured/model payloads.

The source implements device BF16 M128N64 SG4/8 whole-K controls plus explicit
K256/512/1024 multiply-accumulate loops. Frequencies 0/1/2/4 mean no barrier or
`threadgroup_barrier(mem_flags::mem_none)` before every 1/2/4 loop steps.
The F32 cooperative destination is zeroed once and cast to BF16 once, after
all K blocks. Complete blocks use static tensor extents. Partial tails use
dynamic K tensor extents so MPP masks loads; no source conversion or staging
is introduced.

This follows Apple's accumulation synchronization advice in the
[MPP Programming Guide, sections 2.3.4 and 2.3.5](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
and [M5/A19 talk](https://developer.apple.com/videos/play/tech-talks/111432/).
Its application to the current shapes is an experiment. MPP's internal
reduction order is opaque. Full BF16 output parity is measured rather than
promised, and any output change is labeled a numerical alternative.

CPU build/freeze:

```
make -f dev/benchmarks/dense_kloop_sync_sep21/Makefile -j4 all cpu-self-test preview freeze
```

Bounded Root-only first GPU screen:

```
build/dense-kloop-sync-sep21/oracle --run --projection-filter in_proj_qkv --blocks 256,512,1024 --sync-frequencies 1,4 --samples 10 --repeat 4 --out build/dense-kloop-sync-sep21/qkv-v1.json
```

Default covers all seven captured roles, both SIMD group counts, all three K
blocks, and synchronization frequencies 1/2/4. A same-loop no-barrier control
is automatically included for each block/group. `--groups 4` or `--blocks 512`
can narrow the first screen. `--preview` reports exact selected variants from
manifest metadata without reading payloads.

The oracle checks a synthetic K288 tail with partial output intervals and
nine invalid parameter/dispatch cases per pipeline. Actual captures verify
their input/weight/output SHA256, exact selected production output, every full
candidate BF16 output, deterministic repeats, guards/sticky diagnostics,
same-group whole-K parity, same-loop no-barrier parity, and sampled original
BF16 operand FP64 products. The report separately records same-loop output
parity with and without synchronization.

Qualification completes before two final GPU-only warmup rounds. Timings use
balanced AB/BA pairs against selected production, rotate candidate order each
round, and record all GPU/wall/paired samples. No CPU operand/output/guard
access occurs between GPU-only warmup and completion of all timing calls.
Afterward every full output and immutable operand SHA256 is checked again.
Component parity/performance is separate from full-model/service qualification.
