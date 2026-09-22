# Native M64 low-SIMD expert screen

This is a private component. It reads one certified Full512 layer only when
Root explicitly runs it. Source preparation, compilation and the CPU self-test
create no Metal backend and read no model payload.

The control is the qualified M32/SG2/fixed-K128 kernel with its M16 adaptive
tail. Candidates use original native M64 bucket jobs, capacities and grids;
only the gate/down producer shader and thread count change. Both M64/SG4 and
M64/SG2 use ordered F32 K128 accumulation with unchanged I8 codes, F32 row
scales and BF16 projection/SwiGLU boundaries. Every M64 input operand has
static full 64-row extent. The existing +63 global pad covers the last job's
read; independent matrix rows cannot affect another row. Every original
valid-row store mask and original job/rank/range validator remains present.

At 2048 spread rows, forty routes/expert produce 1024 M32 jobs versus 512 M64
jobs. These native job-generation costs and all eleven expert-chain
dispatches are timed. No extra hit-only jobs, coefficient cache or allocation
is introduced. The gate/down traffic saving and a possible20–40 ms whole-model
gain are hypotheses; the earlier neutral I8 M64 experiment used SG8 only.

Before timing, all raw-F32 gate/up/down probes, scaled BF16 probes, complete
activation/down/combine bytes, independent jobs, canaries and sticky
diagnostics must pass exactness. Every eligible variant/control receives at
least 100 ms paired GPU warm work after CPU scans. Timings rotate candidate
position and alternate pair order without any CPU buffer access. Full-model
quality and performance remain separate pending qualification.

```sh
make -f dev/benchmarks/prefill_m64_low_sg_sep21/Makefile -j2 cpu-self-test
.venv/bin/python -B dev/benchmarks/prefill_m64_low_sg_sep21/seal.py
.venv/bin/python -B dev/benchmarks/prefill_m64_low_sg_sep21/run.py --source-check
```

Root must serialize this optional GPU command with all other model/GPU jobs:

```sh
.venv/bin/python -B dev/benchmarks/prefill_m64_low_sg_sep21/run.py \
  --rows 2048 --pairs 8 --pattern spread-all \
  --report build/release/flash/sep21-m64-low-sg-spread-v1.json --run
```
