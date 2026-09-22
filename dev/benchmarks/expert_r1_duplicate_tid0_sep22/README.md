This bounded component moves the thread-zero predicate ahead of the original
duplicate-ID loop. Every thread retains its own-ID and rank load/check; thread
zero retains the same duplicate comparisons and sticky diagnostic writes.
No scan, tensor, descriptor, floating operation, allocation, dispatch, barrier,
or source operand changes. Original GU/down grids are 10×1×10 and 40×1×10,
with 128 threads and M16N64/SG4/dynamic whole K.

The original AIR loads other route IDs before the late thread-zero predicate.
Moving that predicate reduces logical lane evaluations from 576,000 to 4,500
per R1 layer. This is a source count, not physical memory traffic or measured
runtime saving. Compiler and GPU costs must be measured.

Original gathered AIR remains the authoritative control. Explicit restoration
journals must recover the exact original duplicate loop and host harness.
Private helpers are isolated from shipping weak symbols. CPU admission binds
native FP/load/control/attribute and tensor/threadgroup proofs, coupled taps,
and the rank/diagnostic equivalence of the moved predicate.

Root alone reads the ten original expert ranges (49,305,600 bytes) and executes
GPU work under the normal 128 MiB fixture plan. Exact raw/scaled F32 and BF16
projection, SwiGLU, down and own-chain outputs, complete CTA census, duplicate
and malformed diagnostics, immutable operands, canaries, host refusal, healthy
replay, and measured allocation teardown must pass before timing. Both full
two-dispatch variants receive at least 150 ms of GPU warmup and eighteen
balanced sample pairs, without CPU buffer reads during warmup or timing.

The finite-summary and axis-permutation variants remain closed. This CPU
component does not authorize worker integration, standard-model timing, or
promotion; Root's actual component result is required first.
