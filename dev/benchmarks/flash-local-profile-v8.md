# Local v8 defaults

Historical profile: local v9 now adds idle residency maintenance to these
same 37 routes. See `flash-local-profile-v9.md` for current defaults. The
`_local_profile_v8_candidate()` helper preserves this v8 flag set and 192 GiB
hardware gate.

At its qualification, the accepted launcher and `.splash-local-profile.json` matched
`m5-ultra-flash-next-v8` with 37 static defaults. It adds only
`SPLASH_FLASH_MTP_Q8_BF16_REGISTER=1` to v7. The route reconstructs the exact
cached BF16 vocabulary coefficients in registers from the original Q8/G64
bytes for trained joint-MTP `Last` projections with two through four real
lanes. It uses M8N32/K64/SG1 F32 accumulation and existing padding, adding no
weight or scratch allocation. Other row counts, singleton execution, target
vocabulary projection and prefill without logits keep their existing routes.

The full trained-head proof matched every vocabulary BF16 word, premixer word,
QSA plane and greedy result in normal calls, four future continuation steps
and truncate/overwrite branches at contexts 128 and 2048. The service passed all
22 quality/lifecycle cases with all 28 comparison records exact. The matched
ON/OFF benchmark produced identical 21 outputs, draft acceptance, verifier
work, source identities, artifact identities and memory accounting.

The directly affected head decode work fell from 2847 to 2610 ms, an 8.33%
reduction across 1022 calls. All 380 four-lane calls selected the new route,
saving about 0.624 ms each. A repeat ON run retained an 8.07% reduction against
the same OFF control. Short concurrent HTTP throughput improved 4.72% in that
confirmation, but this is sensitive to the slow OFF sample and is not a robust
4–6% whole-request speed claim. The initial 6.01% result included that sample. Long
concurrent HTTP throughput changed only 0.12% in the confirmation. Singleton
and prefill timing differences are drift and are not attributed to this
joint-vocabulary route. See
`build/release/flash/v9-mtp-q8-bf16-register-independent-service-audit.json`
for the measurements and scope.

An explicit zero for `DENSE_CACHE`, `BATCH_MTP`, `MTP` or ordinary `BATCH`
suppresses an implied register default. `DENSE_CACHE` and `BATCH_MTP` are route
construction requirements; `MTP` is a worker requirement. Ordinary `BATCH`
is inherited launcher policy, and explicit joint/register overrides remain
authoritative. The launcher lists transitively implied parents directly because
its merger does not recurse. `INT8_HEAD`, `GPU_GREEDY`, `BATCH_MTP_PREFILL` and
`BATCH_PREFILL` are independent. Explicit child values remain unchanged for
native validation.

All 109 focused profile/launcher CPU tests passed with no skips, including the
native exact-zero saved-store selector check. Historical v5/v6/v7 copies retain
their exact 30/34/36 static defaults. Original source/layout/architecture and
hardware guards, dense/TOP64 pins, store 0 opt-outs, 2048-row windows and
singleton draft depth 15 are unchanged. QSA N32, private loaders, command
retention, TOP128 and GPU prefill copy are not defaults. Original checkpoints,
oMLX preferences and GitHub are untouched by this promotion.

Root verified normal launcher readiness, four complete128-output requests,
95 register-vocabulary commands and scheduler idle recovery on8011. Tracing
is off. The normal binary/metallib match the qualified runtime byte-for-byte.
The profile review preparation and activation performed no GPU commands.
