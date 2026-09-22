# Local Flash-Next profile v11

This is the preceding accepted42-flag depth3 snapshot. Current defaults use
[V12](flash-local-profile-v12.md), which adds only exact public bulk QSA and SG8.

This accepted historical M5 Ultra / 256 GiB snapshot enables
`SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY=1` and
`SPLASH_FLASH_PREFILL_DENSE_TILES=1`. These are the only two additions to the
preceding accepted V10 snapshot: 40 becomes 42 static flags. Draft depth remains
3, the 2,048-row prefill windows remain unchanged, PLE uses SSD streaming with
its native 64 MiB row cache, and the existing qualified Top64 expert/operand
stores retain their original gates. V11 selects no experimental larger expert
inventory, BF16 cache, wide prefill or split-K route.

Teacher priming now computes only the trained head's persistent QSA cache.
The exact existing input projections and cache preparation/pooling remain;
unused attention output and MLP work are skipped. This dedicated operation is
used only for sequential prompt priming. Generic None/Last/All forwards,
speculative proposals and committed folds keep complete computation. True
grouped head priming retains its previous full None forward. Native status
exposes the selected teacher route and successful cache-only call count.

Dense tuning selects measured whole-K BF16 projection tiles for exactly
2,048 physical rows, restricted to authenticated cached main projection roles.
Unknown/MTP/shared-expert roles and other geometries keep previous routes.
The original public BF16 router helper does not select these tiles. Main
layer 1 PLE key was exercised by full-model hidden/logit and HTTP output parity;
it was not one of the ten primitive-captured role fixtures. Native kernel route
identity includes the enabled dense tile policy.

On identical uncached 2K coding prompts, 256 output tokens, one request at a
time, temperature 0 and reasoning off, the same-build three warmed HTTP trials
improved median prefill from **2,097.4 to 2,424.0 tok/s (+15.6%)**. Streaming
decode was **71.60 versus 71.74 tok/s**. Every response matched the original
256-token golden hash, and accepted/proposed speculative counts were identical.
The enabled run completed exactly 16 cache-only teacher commands per request,
64 over warmup plus three samples. This is a prefill improvement; it does not
establish 4K tok/s or a meaningful decode speedup.

Paired OFF/ON service runs passed all eight protocol/lifecycle cases and all
nine non-control normalized responses matched. The independent head oracle
compared 3.10 GB of all five persistent cache planes, future hidden/logits/greedy,
EOS and pooling boundaries, rollback/reappend, generic None contracts and
numeric poison behavior. The same-build uninstrumented native check matched
all three target hidden/logit byte hashes. Sources:

- `build/release/flash/prefill4k-safe-http-performance-v1.json`
- `build/release/flash/prefill4k-safe-service-audit-v1.json`
- `build/release/flash/prefill4k-teacher-cache-oracle.json`

Explicit opt-outs remain authoritative:

```sh
SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY=0 \
SPLASH_FLASH_PREFILL_DENSE_TILES=0 \
./splash serve --model local/Qwen3.8-Flash-Next-oQ4e-mtp \
  --local-package install/local-models/Flash-Next-oQ4e-mtp-v1 \
  --port 8011 --max-context 8192 --no-webui
```

An explicit
MTP=0 suppresses the implied teacher default; DENSE_CACHE=0 suppresses implied
dense tiles. Explicit contradictory child 1 values survive launcher merging
and fail native startup before model/backend loading. Both selectors strictly
accept unset/0/1. Dense role/geometry qualification still applies when enabled.

Only the exact versioned V11/source/architecture/hardware tuple supplies
these defaults. Previous saved profiles do not activate partial V11 settings.
Historical V5–V9 helpers retain 30/34/36/37/38 flags and depth 15. The V10 helper
now represents the actual preceding accepted40-flag depth 3/N32 snapshot.
`_local_profile_v10_placement_candidate` separately preserves the original
39-flag depth 15 placement snapshot. Neither historical definition inherits the
two V11 flags. Optional store paths remain dynamically qualified.

Normal runtime rebuild and ordinary-launch proof completed. Two 2K/256 requests
with no runtime overrides selected 42-flag V11, depth3, teacher-cache-only and
the dense identity tag. Both golden outputs matched; 32 successful teacher
commands completed and native status returned healthy idle with a valid memory
ledger. The single warmed confirmation observed ~2,407 prefill tok/s; this is
not an additional benchmark mean. Four actual invalid/contradictory-selector
launches rejected before backend creation. Proofs are
`prefill4k-v11-default-runtime-proof-audit.json` and
`prefill4k-v11-native-selector-negative.json` under `build/release/flash/`.
