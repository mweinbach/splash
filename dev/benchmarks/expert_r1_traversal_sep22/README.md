This private component tests one change to ordinary R1 I8 expert dispatch:
the candidate passes `uint3(physical.z, physical.y, physical.x)` to the frozen
original native MPP helper. Gate/up remains a 10×1×10 grid. Down uses a
10×1×40 grid in place of 40×1×10. Both variants retain 100 gate/up and 400 down
threadgroups of 128 threads, M16N64/SG4, dynamic whole K, the original tensor
views, and the original late scales and BF16 boundaries.

The hypothesis is that a different assignment of logical output tiles to
physical grid coordinates may improve scheduling or coefficient access.
Metal does not promise an X-major threadgroup execution order. No GPU die
placement, physical bandwidth, universal speedup, or exactness is inferred
from the coordinate bijection.

The authoritative control is the unchanged original gathered AIR
`a0cd35e03daf13324d0308c8b4d8cee1d6d932989be6cbdc0429e6ec471d05c2`.
The candidate's native MPP call and scalar floating-point expression trees
must be authenticated against that AIR. Untimed tap wrappers must match
untapped shipping outputs before and after timing. Root must then qualify
all raw/scaled F32 and BF16 bits, own activation/down chains, original finite
and sticky diagnostic behavior, complete CTA coverage, immutable operands,
canaries, and host alias/extent/full-grid refusal.

Only the Root GPU invocation opens coefficient payloads. It reads the three
original projection ranges and scales for ten distinct experts, totaling
49,305,600 bytes, and packs them into ranks 0–9 without changing coefficient
bits. It constructs no Full512 store or target model. The synthetic finite
BF16 R1 input does not stand in for current causal model inputs. The normal
MemoryGovernor admits 128 MiB before allocation; actual current and peak
allocation, return to the initial ledger, stop, and backend destruction are
measured before success publication.

Gate/up, down on one frozen shared activation, and the two-dispatch own chain
are timed separately. Each performance variant receives at least 150 ms of
GPU warmup and 18 balanced sample pairs. No operand or output buffer is read
by the CPU from the start of warmup through the end of all timing samples.
The final tap replay is bound to the last measured shipping outputs.

CPU preparation does not grant standard-model benchmarking or promotion.
If the bounded component passes and wins, Root must qualify actual current
R1 inputs and full state/output recovery before any isolated standard-only
Teacher composition or same-binary standard timing. The original R1 vector
alternative remains closed after failing current BF16 projection/chain gates.

The older ordinary-AR1 diagnostic stage report assigned 3.77440 ms to gate/up
and 1.49725 ms to down over 48 layers, totaling 5.27164 ms (18.67% of its
28.2340 ms of attributed dispatch time). Those historical measurements
identify gate/up as the larger expert target; they are not current Teacher
timings or a promised whole-model gain.
