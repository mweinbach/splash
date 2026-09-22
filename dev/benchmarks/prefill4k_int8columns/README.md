# Private INT8 expert column reuse

This one-layer primitive tests M32N128SG8 and M32N256SG16 against the actual
persisted INT8 store control. Native M32 buckets, jobs, counts, parameter bytes,
Q4 misses, BF16 activation boundaries, scatter and combine remain unchanged.
There is no extra hit prefix/emission dispatch or coefficient conversion.
Wider output tiles reduce gate/up groups per job10→5/3 and down40→20/10.
The final N256 gate/up tile has128 columns: both B operands carry the actual
logical column extent, with an explicit column mask before scale/output access.

The visible SDK supports these BF16×signed-I8 whole-K shapes. This does not
establish a hardware speedup or numerical result. The executable reflects
requested256/512 threads, pipeline limits and static threadgroup memory first.
Full live BF16 activation/down/combined tensors must be byte-identical to the
current control before timing. Separate uniquely named audit helpers capture
scaled raw F32 gate/up/down values; independent FP64 absolute certificates cover
16 active hit experts, first/final rows and columns crossing all tile boundaries.
Audit buffers have governor admission and guards. Checks stay outside timing.

The default single process constructs/checksums the store once, then tests all
five routing patterns plus an actual-code exact-cancellation input. Ordinary
synthetic BF16 inputs are RMS normalized; they are not captured model states.
An explicit raw input+route-ID pair is supported by the inherited executable
but is not supplied by this wrapper. Source/sidecar immutability and replay after
producer-owner destruction are checked separately. No full-model quality claim.

Build/self-test, without Metal backend or GPU work in the self-test:

```sh
make -f dev/benchmarks/prefill4k_int8columns/Makefile -j2 all
make -f dev/benchmarks/prefill4k_int8columns/Makefile cpu-self-test
```

Only Root may execute GPU screens after excluding every other active model:

```sh
.venv/bin/python dev/benchmarks/prefill4k_int8columns/run.py --pairs 4 --report build/release/flash/prefill4k-int8-columns-layer0-v1.json --run
```

The installed/source-frozen model and existing Top256 artifact remain unchanged.
