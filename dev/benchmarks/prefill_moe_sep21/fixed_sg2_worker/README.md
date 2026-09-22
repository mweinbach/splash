Private incremental prefill scheduling over the exact pointwise/gathered/bulk
Full512 worker. Root's bounded expert-chain component screen measured variant7
M32N64 SG2 fixedK128 at about5.301ms versus5.653ms for the current I8 control,
with complete BF16 activation/down/combine equality. That component evidence
is not actual-model equality or throughput qualification.

```sh
.venv/bin/python dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker/overlay.py
make -f dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker/worker.mk -j3 all cpu-self-test
.venv/bin/python dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker/witness.py
```

Frozen output: `build/prefill-moe-sg2-k128-pointwise-sep21-worker-v1`.
`SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT=7` selects only main nonverification
R2048/M32 canonical Full512 calls. Unset/`0` retains the original graph and
identity; other values reject before metadata/backend/model work. All other
row shapes, decoding, verification and the separate ordinary batch executor
retain their existing BF16-A/I8-B producers. Coefficients, F32 late row scales,
BF16 stage boundaries, native job lists, pointwise combine/poison and trained
MTP weights remain unchanged. No buffer allocation or extra dispatch is added.

The Store constructor and numerical derivative seed remain byte-identical.
An immutable active execution marker distinguishes kernel routes. Dynamic
`fixed_sg2_prefill_route_counters` live outside identity and report graph
construction, not GPU completion. Actual model BF16/logit/token/state equality
must pass before this intended exact scheduling route is promoted; a mismatch
would invalidate that exactness assumption.

The overlay verifies and copies240 inherited source seals, adds three private
files, and freezes117 link inputs. Only Store, Forward and Worker objects are
rebuilt. All inherited objects/AIRs come from explicit `link-inputs.mk` records,
including pointwise objects outside the partial host directory. Builds use the
sealed header closure without a live runtime fallback. The private shader
exports only the two variant7 producers and preserves the qualified shared
primitive arithmetic byte-for-byte.

Compiler/Metal/link, flag0/7 helper and complete worker CPU checks passed.
The seal witness is `cpu-sealed-witness-v1.json` in the build directory. Root
owns every GPU/model execution; this preparation read no model payload.
