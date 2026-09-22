The exact GPU component proof passed at 128 and 2048 token contexts. The
candidate remains experimental: its modest head-only host saving does not
justify changing the production backend and head-state contract in this round.
No HTTP/service or production-default qualification was performed.

The private depth-three candidate chains two unchanged scalar Lightning-head
bodies in one Metal command. The committed fold still produces proposal one;
the chain consumes its greedy token and hidden state to produce proposals two
and three. This preserves the current singleton depth-three numerical route.
The specialization has two bodies rather than encoding a third, skipped body
from the older four-proposal prototype.

Build and CPU-only validation:

```sh
make -f dev/benchmarks/ultra_draft_chain.mk \
  BUILD=build/ultra-draft-chain -j8 ultra-draft-chain-oracle
build/ultra-draft-chain/ultra-draft-chain-oracle --cpu-self-test
```

The build recompiles every host object against the private indirect-dispatch
header. Its ComputeDispatch ABI differs from production: ordinary production
objects must never be linked into this executable. The copied head's original
scalar-forward implementation currently matches runtime/flash/FlashMTP.cpp;
its append bridge adds encoding without token readback, submission, diagnostic
reset, or logical-length publication.

Root serializes GPU work and runs this command with the current local profile
environment, substituting the installed Flash package directory:

```sh
build/ultra-draft-chain/ultra-draft-chain-oracle \
  build/ultra-draft-chain/ultra-draft-chain.metallib PACKAGE \
  build/flash-mtp-gpu-chain-four-v1/prompt128.json \
  build/flash-mtp-gpu-chain-four-v1/prompt2048.json \
  build/release/flash/ultra-draft-chain-depth3-proof.json
```

The GPU oracle compares six alternating paired samples at each real target
context. It checks each executed BF16 hidden row and greedy record, complete
QSA state, rollback continuation hidden/full vocabulary logits, and no-write
behavior when depth, bonus quota, first EOS, or invalid seed suppresses bodies.
Root ran this proof successfully with the current local-profile environment.
Both natural fixtures produced all three proposals, consumed two pairs, and
contained no natural EOS. All listed exact-state and skipped-body checks passed.
Later EOS is covered by the reused CPU policy; forced later-EOS GPU coverage
and HTTP/service qualification remain separate requirements. These fixtures
begin at offsets divisible by four; this depth-three screen does not independently
exercise every QSA compression-block residue or a large context range.

Observed medians over six alternating paired samples per context:

| Context | Baseline API span | Chain API span | API speedup | Command wall speedup | GPU duration change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 tokens | 3.567 ms | 3.410 ms | 4.59% | 4.39% | +2.04% |
| 2048 tokens | 3.786 ms | 3.681 ms | 2.84% | 3.66% | +1.89% |

The API span excludes feature generation, construction, numerical snapshots,
and inspection copies. The measured savings are 0.157 ms and 0.104 ms per
two-body chain, respectively. GPU time increases slightly; the observed saving
comes from reducing host/command boundaries. This is not a measured HTTP gain.

The current depth-three singleton baseline trace contains 340 draft-body
commands across 170 cycles: 0.564 seconds GPU, 0.633 seconds command wall. Each
two-body cycle has median 3.316 ms GPU and 3.718 ms command wall; median host
gap between bodies is 0.120 ms. Target verification consumes 5.529 seconds GPU
and 5.676 seconds command wall, about 85% of traced decode GPU time. Applying
the observed per-chain host savings uniformly to the 170 baseline cycles gives
an illustrative 18-27 ms saving, roughly 0.3-0.4% of traced decode command wall
time. That model is workload-specific and is not end-to-end validation.

Production integration is deferred. A future integration would require:

- Promote the registered indirect-source API and backend validation/retention/
  barriers. Rebuild all host objects and retain profiling metadata for indirect
  dimensions. GPU-owned dimensions must have no host readback.
- Share the existing scalar head graph builder between forward and chaining,
  rather than maintaining copied arithmetic. Publish the head logical length
  once after the command completes and poison state after numerical body or
  command failure. Retain capacity-only rollback behavior.
- Add one bounded singleton Worker branch for eligible greedy depth-three
  requests with no waiting peers. Keep the current route for joint lanes,
  depth zero/one, other requested depths, and disabled GPU greedy.
- Preserve target verification/acceptance and head truncation. Record the two
  consumed head pairs and three proposals accurately; preserve command and
  phase counters. A safe point runs after the chain; cancellation no longer
  occurs between these two bodies and therefore needs lifecycle validation.
- Compare same-binary on/off whole requests, exact output/state continuation,
  cancellation, terminal output quotas, EOS, cache offsets, and healthy idle
  status. A component exact pass is not service qualification.

This change does not steer allocations to a particular GPU die or change
speculative acceptance. Its proposed benefit is removing a dependent CPU/GPU
round trip while retaining the same arithmetic and proposal sequence.
