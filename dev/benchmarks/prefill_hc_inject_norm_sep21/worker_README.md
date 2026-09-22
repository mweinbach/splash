# Private exact prefill HC worker

This composes the Root-qualified HC inject/norm helper over the sealed pure
Full512 SG2/tail/FMA worker with its independent optional dense-W8 control.
`SPLASH_FLASH_PREFILL_HC_INJECT_NORM_SEP21=0|1` defaults to zero. Zero preserves
the parent's graphs; one selects only singleton main non-verification rows
512–2048. Attention injection produces the immediately following MLP norm.
MLP injection produces the next attention norm only when no PLE update intervenes
and the next consumer is not the terminal mixer. The existing small-row terminal
fusion and audited norm selection stay unchanged. Every public HC-down/up
small-row guard stays unchanged.

The exact Root-qualified shader AIR and bridge are retained verbatim. All
actual parent object/AIR inputs come from its expanded Make target prerequisites,
including dense `worker_cache.o` and `dense-w8a8.air`. Only Forward/Worker objects
are replaced, the attribution source gets strict flag initialization, and one
qualified AIR is added. There is no extra GPU workspace, immutable buffer,
residency request or planner adjustment. All parent headers and transitive
dependencies are retained and verified. FlashTensor declarations and qualified
HC/ABI headers are pinned.

The target numerical derivative expression remains byte-identical to the
parent. An independent frozen flag marker appears in kernel routes; cumulative
encoding counters are separate status fields and never enter identity. The
qualified sticky OR4 nonfinite diagnostics are retained; finite arithmetic is
unchanged. Counters observe encoded graphs, not successful command completion.

CPU compilation, selector/scope/identity tests, startup rejection before paths,
source/link/header closure and worker CPU tests must pass before Root runs the
whole-model benchmark and 22-case semantic suite. The component's strict GPU
certificate covers synthetic inputs; whole-model parity and throughput remain
pending. Root alone runs GPU/model jobs and serializes them.

Prepare and compile (CPU only):

```sh
.venv/bin/python -B dev/benchmarks/prefill_hc_inject_norm_sep21/worker_overlay.py
make -f build/prefill-hc-inject-norm-sep21-worker-v3/machinery/worker.mk BUILD=build/prefill-hc-inject-norm-sep21-worker-v3 -j4 all cpu-self-test
.venv/bin/python -B build/prefill-hc-inject-norm-sep21-worker-v3/machinery/worker_witness.py --build build/prefill-hc-inject-norm-sep21-worker-v3 --output build/prefill-hc-inject-norm-sep21-worker-v3/cpu-witness-v1.json
```

For Root's established `splash_tuning_sep21.py` launch, use this worker binary,
set `SPLASH_FLASH_PREFILL_HC_INJECT_NORM_SEP21=1`, and preserve the independently
chosen SG2/tail/FMA/dense-W8/pointwise/Full512/gathered/bulk controls. Use a
fresh output and max-context 16384 for the semantic plan. Run flag zero in the
same binary as a matched control. Model/semantic readiness requires Root evidence.
