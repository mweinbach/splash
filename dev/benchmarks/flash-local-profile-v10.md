# Local Flash-Next profile v10

This is historical documentation. Current defaults use
[V11](flash-local-profile-v11.md), with42 flags/depth3. The accepted V10
snapshot described first below has40 flags/depth3/N32 on; the original
placement-only39-flag/depth15 snapshot and its measurements are retained
separately below.

This preceding accepted V10 snapshot defaults to MTP draft depth 3 (changed September
21). Historical qualification results below used depth 15. Explicit
`SPLASH_FLASH_MTP_DRAFT_DEPTH` environment overrides still take precedence.

The same update enables `SPLASH_FLASH_QSA_OUT_F32_N32=1` for the already
source-qualified Q5/Q6/G64 main QSA output projections at 4–16 rows. The preceding accepted
profile has 40 static flags. The exact output projection route and cached
CPU status percentiles passed matched model-output and service checks; observed
full-request decoding gains were under 1%. Historical results below used the
original 39-flag profile. Dense traversal remains opt-in, and split-K and draft
chaining remain isolated experiments. See [results](ultra-locality-results.md).

The qualified M5 Ultra /256 GiB local profile now uses SSD streaming for the
PLE n-gram table. The sole static change from v9 is
`SPLASH_FLASH_PLE_SSD_STREAMING=1`, bringing the profile to39 flags. All kernel
routes, idle residency maintenance, source identity and optional saved operand
store gates remain unchanged.

The native row cache defaults to64 MiB. The profile does not set
`SPLASH_FLASH_PLE_SSD_CACHE_MB`, so opting out cannot leave an implied cache
setting that conflicts with resident mode. Explicit environment choices remain
authoritative; explicit CLI placement choices override both the profile and
environment.

Normal startup uses SSD streaming automatically:

```sh
./splash serve --model local/Qwen3.8-Flash-Next-oQ4e-mtp \
  --local-package install/local-models/Flash-Next-oQ4e-mtp-v1 \
  --port 8011 --max-context 8192 --no-webui
```

Add `--no-ple-ssd-streaming` or set `SPLASH_FLASH_PLE_SSD_STREAMING=0` to
restore the resident table. Add `--ple-ssd-cache-mb 128` to change the SSD row
cache after the effective profile has enabled streaming. A cache size while
streaming is off is rejected before server execution. Remote models cannot
use either placement CLI option. Unknown local packages/hardware receive no
SSD default and require explicit `--ple-ssd-streaming` to opt in.

Historical review helpers preserve their exact static settings: v5=30,
v6=34, v7=36, v8=37, v9=38. Their source and memory gates remain distinct:
v5–v8 use192 GiB, v9/v10 use256 GiB. Historical on-disk profile copies cannot
activate partial current defaults.

The [SSD qualification](flash-ple-ssd-streaming.md) includes22 SSD and22
resident service checks, exact paired outputs,29.8 GiB lower retained native
peak allocation, and matched warmed HTTP throughput within1%. Those existing
measurements qualify the route; launcher CPU tests verify default selection,
explicit opt-out, cache validation and unchanged historical gates. This
profile change does not change weights or kernels.

Normal launcher proof completed with no SSD opt-in argument: SSD mode and the
native64 MiB budget selected automatically, single and four-request128-output
waves passed, and the service returned healthy idle. A request after9 seconds
idle reached first content in413 ms versus370 ms immediately, with identical
16-token output. Retained proofs are `default-v10-normal-smoke.json`,
`default-v10-ssd-idle-proof.json` and `default-v10-final-idle-status.json` under
`build/release/flash/`. All139 focused launcher/profile tests passed, no skips.

See [workspace guide](../../LOCAL_WORKSPACE.md) for active files, cleanup
scope and retained evidence. Source/model stores/reports are preserved;
stale compiled experiments and bulky raw profiles were removed locally.
