# M8/N64/SG4 native expert INT8 feasibility

Scope: CPU/source and compiler only. No GPU commands, model payload reads, or production edits.

## Result

The existing native expert INT8 gate/up and down templates accept `M=8, N=64, SG=4` and compile/link successfully with the current production hybrid flags. The same 128-thread dispatch is consistent with their existing `threads.x == SG * 32` guard and the SDK requirement that dispatched SIMD-group count match `execution_simdgroups<4>`.

M8 versus M16/M32 exact floating accumulation order is **not established** by source/compile evidence. All three lower to the same externally implemented MPP BF16 x signed-INT8 -> F32 operation with different matmul descriptor M. The local SDK does not expose the K reduction loop or guarantee bitwise equality across descriptors. A GPU primitive oracle must certify equality before dispatching M8 as an exact candidate.

## Source facts

- `runtime/metal/kernels/shared/flash_int8_expert_store.metal:293` (`int8_expert_store_gate`) and `:339` (`int8_expert_store_down`) use ordinary device tensors for BF16 left and signed INT8 right operands, F32 cooperative destinations, dynamic K, `transpose_left=false`, `transpose_right=true`, `relaxed_precision=false`, and multiply mode.
- They mask incomplete expert buckets using left tensor `valid_rows=min(M,end-begin)` and suppress stores outside valid rows. M8 therefore preserves this visible bounds behavior without staging.
- Existing M16 and M32 instantiations both use SG4. Only M64 currently uses SG8.
- The SDK data type table explicitly includes `bfloat, int8_t, float` in `MPPTensorOpsMatMul2d.h:38`.
- SDK scope documentation `MPPTensorOpsMatMul2d.h:319-339` says all threads in the execution scope must enter `run`, and mismatched dispatched/configured SIMD-group count is undefined behavior.
- SDK SIMD-group(s) assertions in `__impl/MPPTensorOpsMatMul2dImpl.h:6370-6396` require M and N multiples of 8 or 16, at least one of M/N multiple of 16, and static K a multiple of 16 for non-subbyte types. M8/N64/dynamic-K passes all of these.
- The single-SIMD-group restriction and M/N/K 16-or-32 restrictions at `:6320-6367` apply to cooperative **input** tensors, not this ordinary-device-input/cooperative-destination path.
- `relaxed_precision=false` avoids permission to sacrifice accuracy for performance (public header `:120-121`), but the SDK does not document a fixed accumulation order for it.

SDK headers were read from:
`/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/MetalPerformancePrimitives.framework/Versions/A/Headers/`.

## Compiler evidence

A temporary source cloned the production kernel and added only these two instantiations:

```metal
INT8_GATE(flash_int8_expert_store_gate_up_m8_n64, 8, 4)
INT8_DOWN(flash_int8_expert_store_down_scatter_m8_n64, 8, 4)
```

Temporary directory:
`/var/folders/y_/z9xdzx553g59hfq71d6tw1gw0000gn/T/splash-m8-feasibility-zfdep4s6/`

Compilation passed with no diagnostics:

```sh
xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime \
  -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 \
  -c <temp>/candidate.metal -o <temp>/candidate.air
xcrun -sdk macosx metallib <temp>/candidate.air -o <temp>/candidate.metallib
```

A second `metal -S -emit-llvm` compile passed. Its AIR IR contains the M8 entrypoints and calls
`__tensorops_impl_matmul2d_op_run_cooperative_dv_b16_dv_i8_f32_v2` for M8/M16/M32. Descriptor parameters and cooperative tensor capacities are external implementation inputs; the IR does not reveal the reduction order.

No performance or runtime pipeline-creation result is claimed.
