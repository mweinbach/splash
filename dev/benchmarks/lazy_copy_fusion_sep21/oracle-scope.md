# Exact lazy copy fusion qualification

`oracle.mm` is a private, synthetic layer qualifier. It has been checked with
Clang `-fsyntax-only -Wall -Wextra -Werror` against the final candidate headers.
No GPU result is implied by that check. `--cpu-only` returns before construction
of MetalBackend or access to any model payload.

The unchanged frozen runtime record is constructed with
`SPLASH_FLASH_GDN_LAZY_COPY_FUSION_SEP21=0`; the candidate is constructed with1.
Both retain their route selection after the environment changes. The candidate
loads raw projected QKV into the exact active view returned by its own record.
An independent captured Persistent512 graph supplies retained-prefix states.

Hard gates require zero differing bytes for raw/projected operands, saved
mixed/decay/beta, initial history and FP32 state snapshots, BF16 recurrence rows
and output, full live history/state with padded strides, diagnostics, every
inactive arena byte and allocation guard. Snapshot and prepared operand bytes
must remain unchanged through commit and future continuation.

The default matrix covers R1/2/3/4/8/16 and lanes1/2/3/4, cold and carried state,
normal and extreme finite coefficients, three changed carried sequences, every
kept prefix0..R, mixed partial/full/terminal lanes, completed abort, and changed
ordinary future continuation of1/2/4 rows. Wider R8/R16 joint windows are skipped
because native joint verification admits at most4 rows per lane. R1 bypass is
qualified through the unchanged native graph.

Host negatives cover invalid geometry, selector text, epsilon/strides/extents,
foreign and overlapping RawQKV views, exact-view shape mismatch, all other tape
aliases, pending/replacement/stale/foreign tickets, and zero-allocation R1 tape
planning. Sixty-four raw numeric cases cover signed zeros, subnormals, minimum
normal, NaN and infinities at eight input/coefficient/state boundaries. They
preserve the native diagnostic behavior; some sigmoid/softplus infinities map
to finite values in the original kernel, so this does not invent a universal
nonfinite-input rejection requirement.

Run only with an immutable linked build and source/object/library provenance:

```
lazy-copy-fusion-oracle --cpu-only
lazy-copy-fusion-oracle FROZEN_METALLIB FRESH_REPORT_JSON
```

Environment selectors `GDN_LAZY_FUSION_ROWS`, `GDN_LAZY_FUSION_LANES`,
`GDN_LAZY_FUSION_COLD`, and `GDN_LAZY_FUSION_EXTREMES` accept CSV integer lists.
`GDN_LAZY_FUSION_NUMERIC=0` deliberately skips the separately reported numeric
matrix. Every GPU submission validates the200-byte CommandTiming ABI and a
finite duration strictly between1e-9 and600 seconds. This oracle makes no
performance/timing qualification.

Build closure requires copies of this source, `gdn_fixture.hpp`, transitively
included frozen runtime headers, linked object bytes and their frozen source
provenance, and the actual metallib. Generate
`LazyCopyFusionOracleBuildProvenance.hpp` with
`kLazyCopyFusionOracleBuildProvenance` set to the closure JSON. The diagnostic
fixture is an exact source copy of the existing scalar GDN test with its main
renamed during inclusion. The runtime object subset is FlashGDNLazyRollback,
FlashGDN, FlashGDNFused, FlashGDNSeparate, DeviceCapabilities, and MetalBackend.

This proves neither whole-model logits/QSA cache equality nor trained MTP
acceptance or numerical-quality-suite success. A separate whole-Forward oracle
is prepared in `whole_oracle.mm`; its result must be checked independently.
