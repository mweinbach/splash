# PLE direct source lookup

`SPLASH_FLASH_PLE_LOOKUP_FUSED=1` opts the text trunk into one immutable Metal
Tier 2 argument buffer for the original 128 affine Q4/G32 source shards. The
checkpoint tables and coefficients are neither copied nor transformed. The
source trunk shares its argument buffer with AR batch, batch verification and
batch prefill. Unsupported Tier 2 hardware uses the ordinary source gather and
reports that fallback in `kernelRoutes()`.

For each actual token and embedding head, thread 0 of one threadgroup computes
the exact stored hash and broadcasts the resulting I64 index. The other threads
read the selected shard directly and reconstruct its 160 BF16 values. This
replaces hash, ordered history update and sixteen eight-shard gathers with one
hash/gather plus one ordered history update. Threadgroups belonging to unrelated
shard ranges are no longer launched.

The hash is unsigned multiplication with modulo-2^64 wraparound and XOR, then
reinterpretation as signed I64 before positive modulo by that head's stored
vocabulary size. Heads 0..7 use the bigram; heads 8..15 use the trigram. EOS itself
retains its prior context; the token following EOS masks older context to EOS.
History remains read-only throughout the fused dispatch and is updated only by
the next ordered dispatch. The existing speculative prefix restoration remains
unchanged.

Affine reconstruction retains the qualified boundaries:

```
row_value = BF16(F32(code) * F32(stored_scale) + F32(stored_bias))
output    = BF16(F32(row_value) * F32(stored_shared_scale))
```

F32 contraction and reassociation remain disabled. Invalid IDs produce BF16
NaNs and sticky diagnostics before any table access. Source pointer arrays are
indexed W 0..127, scales 128..255 and biases 256..383; homogeneous source row
strides remain ordinary dispatch metadata.

The backend reflects the entire read-only pointer layout before accepting an
argument buffer. It rejects incomplete/duplicate/non-dense bindings, foreign
backend buffers, misaligned or truncated views, nested argument buffers and
sparse resources. Shared and Private ordinary views are allowed. Indirect base
allocations are retained by the immutable argument buffer and command ticket,
and their residency/read usage is declared in one deduplicated `useResources`
batch per dispatch. Fixed workspace admission reserves 16 KiB before source
construction; the reflected byte extent must fit that bound.

The initial Root-run v1 oracle passed all 27 synthetic and original-checkpoint
cases with Metal shader validation: exact IDs, gather BF16 bits, final token
history, sticky diagnostics, untouched token inputs and ID/output canaries.
It also covered every source shard's first/last row and padded/nonzero source
views. Its primitive timings were measured **with shader validation** and are
not whole-model performance claims. Evidence:
`build/release/flash/ple-fused-primitive-v1.json`.

The current v3 oracle additionally requires complete reflected bindings, foreign-backend
rejection, mixed Private/Shared source views and survival of source/operator
destruction while an asynchronous command ticket is outstanding. The oracle is
Root-run only:

```sh
MTL_SHADER_VALIDATION=1 FLASH_PLE_FUSED_REPEATS=4 \
  build/flash-ple-fused-v3/flash-ple-fused-oracle \
  build/flash-ple-fused-v3/splash.metallib \
  build/release/flash/ple-fused-primitive-v3.json \
  install/local-models/Flash-Next-oQ4e-mtp-v1
```

The Root-run v3 report passed all 29 cases with shader validation, including
the additional binding, mixed-storage and pending-ticket ownership checks.
Evidence: `build/release/flash/ple-fused-primitive-v3.json`. Whole-model and
uninstrumented service performance qualification are separate checks.

`flash-ple-fused-oracle --cpu-self-test` creates no Metal backend. The 34 CPU
PLE tests include fifteen new integer-contract tests. They independently cover
signed overflow and INT64_MIN, EOS/streaming boundaries, malformed metadata,
diagnostic priority and every source shard address boundary. They also prove a
possible exact reciprocal reduction using multiply-high and one correction;
that math alternative is not selected by this implementation.

`flash-ple-fused-oracle --abi-self-test <metallib>` inspects the actual Metal
reflection, encodes a tiny complete source-pointer table and checks malformed
bindings with zero GPU submissions. The v3 ABI check passed with shader
validation. The initial v2 reflection preflight rejected `metal::array` because
Metal reflects it as a struct wrapper containing `__elems`; v3 recursively
flattens those relative argument IDs while retaining the complete readonly
pointer count and alignment checks.
