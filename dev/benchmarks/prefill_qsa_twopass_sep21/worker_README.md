# Private packed-V QSA singleton integration

Root-qualified component:6.871→3.32665ms inclusive, F32 global probabilities,
whole-K QK/PV, exact prefix/cache/future append checks and strict F64 gates.
The direct-V layout failed numerical checks and is never selected here.

Base is HC worker v3. Strict selector defaults0 and freezes before paths or
backend creation. Flag0 keeps base graphs, numerical identity and allocation.
Flag1 affects only singleton main fresh2048 non-verification attention.
Historical prefixes, decoder, verifier, batch and trained MTP retain their paths.
Existing31,457,280-byte prepared planes are reused. One478,150,656-byte arena
contains aligned Q-pack, score/P union and raw-output views; packed V reuses
dead Q-pack after QK. The planner charges it before the existing real governor
reservation; construction checks the exact one-buffer allocation delta.
External destinations reject the arena, views and prepared planes.

The source-bound global-F32/whole-K policy changes the numerical derivative
only when selected. Mutable encoding/construction counters stay outside
identity. Forward mutex, synchronous GPU completion, numeric poison and Worker
cancellation/deadline safe points stay intact. Whole-model22-case semantics,
actual admission and cancellation/deadline recovery remain Root verification.

CPU-only build:

```sh
.venv/bin/python -B dev/benchmarks/prefill_qsa_twopass_sep21/worker_overlay.py
make -j6 -f dev/benchmarks/prefill_qsa_twopass_sep21/worker.mk
.venv/bin/python -B dev/benchmarks/prefill_qsa_twopass_sep21/worker_witness.py --output build/prefill-qsa-twopass-sep21-worker-v1/cpu-witness-v1.json
```
