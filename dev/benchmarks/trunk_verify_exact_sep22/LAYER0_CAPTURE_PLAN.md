If compact-only isolation fails, keep the combined qualification failed and
prepare a separate, untimed private layer-0 capture oracle.

Run the canonical real 2K Prefill and the same four control verification tokens.
GPU copy-only taps preserve actual active values before shared scratch reuse:

| Location | Capture |
|---|---|
| After layer-0 attention HC/GDN and MLP HC | Actual BF16 mixed input `[4,2560]` |
| After router | BF16 router logits, I64 IDs `[4,10]`, BF16 route weights |
| After expert gate/up | Canonical BF16 activated intermediate `[4,10,640]` |
| After expert down | Canonical BF16 expert down `[4,10,2560]` |
| After combine/injection | Shared down, combined branch, and residual hyper |

Control/candidate must compare these actual inputs before replaying a producer.
Use the existing qualified low-footprint layer-0 control/native six-stage
probes on the captured mixed input and IDs. Capture exact signed I32 dot,
late F32 row scale, gate/up BF16 and SiLU/product boundaries, down raw/scaled
F32, and final BF16. Do not substitute synthetic normalized inputs.

Extra detached capture buffers receive explicit governor admission and guards;
no arithmetic, state, or source-weight mutation. Compare all active words and
check readonly inputs, inactive producer regions, and redzones. Timing and
promotion are outside this diagnostic. Root alone reads/hashes payloads and
executes GPU work. Preparers inspect source/metadata only.
