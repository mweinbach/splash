Bounded standalone online QSA source plan; no worker integration.

Authoritative current parent: build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2.
Executable663663067a6b696811980c5afa3d2cca2dd1b0b28629e6d9b326a7973d084438;
library7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8.
Literal source9f5a1da89e05a20b109330eaced63a645000a5728538c8e9cafc738ce9ae0905;
shipping online AIR7e560b9b3e4c598fa04d3d061ec88b93ca20d705849b550e6390158d97a352d2.

Native and candidate retain original early3328x1x1/128-thread, temporal4608x1x1/
256-thread, reduce2048x24x1/256-thread dispatches. Only GLOBAL V gathers change:
values[(token*2+kv)*256+dimension+d] becomes
values[(kv*256+dimension+d)*2048+token]. Original TG V bank, tensor shape/stride,
K64 QK four partials/scalar additions, eight-lane XOR max/sum, online64-token
recurrence, PV four partials, alpha update and ordered reducer/BF16 gate stay
literal. Both private copies omit only the unused generic non-SG8 entry wrapper;
the source journal exactly reverses that omission, namespace isolation and all
three address replacements to the entire authoritative source. This omission
does not change any selected entry body. Every user helper/type is isolated.

Pack is an independent ushort32x32 bit transpose with padded TG32x33 storage,
one uniform barrier,256 threads,grid64x8x2. It produces exactly2MiB logical
packedV. Host must reserve actual charged owner/canary bytes before allocation,
require fresh main nonverification begin0/rows2048/capacity>=2048/originalSG8,
validate source>=2MiB/packed>=2MiB and all metadata/extents/disjointness/topology,
then order original prepare/cache append ->pack ->early ->temporal ->reduce.
Pack never changes source/cache/selection data and does not scan nonfinite input.
Shader malformed pack shape/topology sets existing QSA bit9 before any reads or
writes. Consumers retain literal original guards and diagnostics. Host refuses
partial/excess original or candidate grids before submission.

Names: qsa_online_packed_v_{native,candidate}_{early,temporal,reduce};
qsa_online_packed_v_{native,candidate}_reduce_tap; qsa_online_packed_v_pack.
Attention buffers0Q,1K,2V,3selected,4F32statistics,5F32numerators,6diag,
7FlashQSAFastParams. Reducer buffers0statistics,1numerators,2Qprojection,
3BFoutput,4diag,5FlashQSAFastParams. Untimed reducer tap additionally binds6
raw F32 quotient[2048,24,256] and7 rounded ungated BF16[2048,24,256]; ONE live
value/sum division feeds both raw store and original BF16 cast/gate. The same
existing BF16 attention SSA value supplies the rounded store. Producer F32 sheets already provide coupled taps;
no extra producer clones or dot recomputations. Pack buffers0ushort sourceV,
1ushort packedV,2diag,3QSAOnlinePackedVParams{2048,2,256,0} (16bytes).

Compile native/candidate/pack/nativeReduceTap/candidateReduceTap in separate
translation units, retaining original safe pragmas and exact authenticated
Metal4.1/O3/default-fast recipe. Separate TUs prevent the final reducer clang-FP
off pragmas from leaking into the other copy's earlier producer helpers.
Actual AIR/IR audit must admit only the explicit V-address map and untimed live
quotient output sink; FP/load/pointer/TGstore/constructor/call/control/attributes
otherwise remain original. Source identity is not GPU parity or a universal
opaque MPP order proof. Root full active F32 stats/numerators/raw quotient/BFout,
physical cache/future/diagnostic/canary/state gate precedes timing. Performance
control remains immutable original shipping AIR, not private native aliases.
Inclusive timing is original3 versus pack+original-shaped3, >=150ms GPU warm
EACH variant,18 balanced pairs, no CPU buffer access in the warm/timed window.
