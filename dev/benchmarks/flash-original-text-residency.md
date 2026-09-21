# Original text residency experiment

`SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT=1` is an opt-in experiment and remains
disabled in the local v5 profile. It changes residency requests only; original
weights, coefficient math, GPU kernels, and model state are unchanged.

The aligned bundle contains 21 original native base allocations. Tensor views
share their owning base. A residency request deduplicates by the native
`MTLBuffer` object and requests its entire allocation; a narrow tensor view
cannot exclude adjacent PLE or vision data.

CPU manifest audit selected payloads 9–21: **13 bases, 66,060,288,000 bytes
(61.523 GiB)**, containing 420 text/MTP tensor views. Every original PLE tensor
occupies bases 1–8; every vision tensor occupies base 1. Those eight bases are
excluded entirely, including their 6.815 GiB of otherwise useful text/MTP data.
The excluded PLE lookup alone is 29.802 GiB. There are no logical tensor overlaps.
With the existing derived mappings and rank tables, the nominal combined set
is 104,066,711,552 bytes (96.920 GiB). These are CPU file/base lengths; actual
native `allocatedSize` totals are reported only by the runtime registration.

`FlashWeights::checkedOriginalTextResidency()` classifies every original tensor
during ordinary metadata load, then omits any whole base containing PLE,
vision, or an unknown tensor family. It returns existing immutable buffer
owners with no new mapping or allocation. The getter requires the exact
qualified source/layout fingerprint, model geometry, 13 selected bases,
66,060,288,000 selected bytes, and eight excluded PLE/one excluded vision bases.
It does not rescan large payloads.

The worker appends the original subset to the final saved-operand union before
the backend's single startup residency request. The backend prohibits a second
registration and does not charge existing backing twice. If original residency
is requested, a governor snapshot must show a valid host measurement, normal
growth/pressure, and headroom for the selected raw bytes plus a 2 GiB margin
above the existing reserve. This preflight check adds no allocation reservation.
If the margin fails, saved-only residency remains available and status reports
why the raw addition was withheld.

`/status.original_text_residency` reports selected paths, mapped bytes, excluded
base counts/bytes, host preflight values, and whether original buffers joined
the union. Actual native union counts/bytes come from the one residency lease.
`physical_pinning_verified` remains false: residency requests accessibility and
does not establish a physical pinning guarantee. `active` requires a successful
lease, a healthy backend, and a running service.

Fresh isolated build: `build/flash-original-text-residency-v1/splash-flash` with
its adjacent metallib. The 44 worker CPU checks and 166 standalone policy checks
passed; policy checks also passed ASan/UBSan. GPU startup/correctness and matched
Off/On callback/GPU/HTTP comparisons belong to Root.

Root completed all 22 quality/lifecycle checks successfully. The matched
experiment showed no useful HTTP gain: medians were 101.0/153.0/64.4/77.9 versus
the refreshed v5 baseline 99.1/151.6/64.6/80.0 for short singleton/concurrent and
2K singleton/concurrent cohorts. The flag remains off by default; no further
GPU qualification is required for this negative result.

Reproduce the CPU allocation audit:

```sh
.venv/bin/python dev/benchmarks/audit_flash_original_residency.py \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/original-text-residency-manifest-audit.json
```
