# Private prefill HC residual injection and normalization fusion

This isolated component copies the literal arithmetic in the existing qualified
small-row `flash_hc_fused_inject_norm` kernel. A separately named host helper and
shader allow only R512–2048/H2560/S4. The public HC-down/up and all decode/MTP
geometry guards remain unchanged. There is no model payload access, new runtime
workspace or production source edit in this component.

Each `(row,stream)` group owns its 2560 BF16 elements, caching four adjacent
injected BF16 values per each of 640 threads. BF16 product and residual addition
remain separate; norm uses the same F32 square traversal, two SIMD reductions,
precise rsqrt, F32 scale formation and final BF16 store. Contraction and
reassociation remain disabled. Hyper update may use exactly the input view;
partial overlaps are rejected using Shared-view ranges. Normalized output and
diagnostics are disjoint. Norm weight is exactly BF16 or F32[10240], with both
one-plus-weight and direct-gamma conventions tested.

The oracle compares native injection+norm against private fusion, byte for byte,
both in and out of place. It covers R512/1024/2048, both weight formats and
conventions, rounding/cancellation, signed zero, subnormal, large finite and
NaN/Inf cases. Stored nonfinite updated/normalized values OR sticky diagnostic
bit 4. Finite square overflow must not introduce extra diagnostics. Invalid
shader parameters or launch sizes OR bit 2 and leave outputs unchanged. Host
alias/extent validation and output canaries are checked.

Timing uses unchanged out-of-place synthetic inputs. Every case receives at
least 150 ms of GPU work before an even number of balanced AB/BA pairs; there are
no CPU tensor accesses between the first warm command and final measured pair.
Native control timing includes its two dispatches. Fusion timing includes its
additional sticky diagnostics. This is a component result, not whole-model
qualification or proof of the 4K prefill target.

Root alone may execute the GPU command, after other GPU/model jobs are stopped:

```sh
.venv/bin/python -B dev/benchmarks/prefill_hc_inject_norm_sep21/run.py --report build/release/flash/sep21-prefill-hc-inject-norm-v1.json --pairs 8 --run
```

Omit `--run` for a CPU-only provenance check and command preview. The runner
verifies every sealed source/artifact hash and refuses to overwrite a report.
`source-manifest.json` records immutable source copies, compiled artifacts,
the CPU self-test and compile-only status. The sealed executable/library live
under `build/prefill-hc-inject-norm-sep21/sealed/`.

Eligibility in a later model overlay must preserve immediate-next-norm ordering:
the MLP→next-HC route excludes an intervening PLE update, and the terminal mixer
retains its audited norm selection. This component has no such model overlay.
The stage-profile bound is small: old 97 norm dispatches totaled 9.61 ms and 96 inject
dispatches 8.47 ms on a 2K prefill. A 6–9 ms saving is a hypothesis until Root measures
this component and a separately qualified whole-model overlay.
