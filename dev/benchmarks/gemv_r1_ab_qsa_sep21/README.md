The sealed worker at `build/gemv-r1-ab-qsa-sep21-worker-v1` composes the existing
ordinary AR1 vector producer onto the exact A/B merge + packed-QSA worker.
The old numerical alternative shipping source and RN/RTZ/FTZ certificate are
copied byte-for-byte, preserving identity `bd384554…f29b`.

```sh
.venv/bin/python dev/benchmarks/gemv_r1_ab_qsa_sep21/prepare.py
make -f dev/benchmarks/gemv_r1_ab_qsa_sep21/worker.mk -j6 cpu
.venv/bin/python dev/benchmarks/gemv_r1_ab_qsa_sep21/witness.py \
  build/gemv-r1-ab-qsa-sep21-worker-v1
```

Strict `SPLASH_FLASH_GEMV_DECODE_R1_SEP21=0/1` defaults disabled and requires
gathered MPP enabled. Only Worker’s ordinary singleton autoregressive call uses
`forwardDecode`. The existing `forward()` entry, one-row prefill/tails, MTP
seeds, verification, trained head and batch executors remain unchanged. The
selector additionally requires physical R1, nonverification and the explicit
ordinary-decode argument. R4 vector execution is excluded.

The source closure contains288files and123frozen linked inputs. An exact
compiler dependency census of48sealed translation units finds six consumers
of modified Forward/Store class headers; all six rebuild and older ancestor
copies are excluded. MTP/BatchMTP do not consume those headers and retain their
old source/policy. All49effective host objects link once, plus four core
objects. No live runtime/benchmark header dependency or new GPU allocation is
introduced. AB math/cache planners and packed-QSA source/fields are preserved.
The alternative adds its known certificate/kernel marker to target identity;
changing graph counters remain outside immutable identity.

The compiled policy processes each pass approximately49KCPU checks, covering
rows1..8192, exact optional-switch/freeze/retry semantics and missing versus0.
The complete Worker CPU suite and six fail-before-path malformed-flag cases
also pass. CPU work touches no GPU or coefficient payloads. Tokenizer-only
driver preparation confirms the canonical2K prompt and four planned waves.

Root’s prepared standard benchmark +22-case semantics command is:

```sh
sh build/gemv-r1-ab-qsa-sep21-worker-v1/run-root-standard.sh
```

It uses one warmup, three measured256-token coding trials, max context16K and
port8035, with the same explicit environment as the completed AB/QSA standard
plan plus vector flag1. Exact argv/environment are in `root-standard-command.json`.
The optional same-binary flag0 attribution control is prepared in
`run-root-flag0-control.sh`; it has a separate fresh report. Root alone runs
these GPU/model commands after scheduling other GPU work.

The earlier R1 worker showed a measured standard decode improvement, but that
does not establish a gain or preserved semantics with the new prefilling state.
Complete generation, semantics, lifecycle and state qualification remains
pending for this composition. Its full model result cannot be inferred from
component quality or source closure checks.
