# Historical local v5 defaults

The v5 profile is retained as tuning history; active defaults are now v7,
documented in `flash-local-profile-v7.md`. The former v5 profile's
30 static environment defaults add `SPLASH_FLASH_MOE_DIRECT_A=1`,
`SPLASH_FLASH_QSA_ROW_TILES=1`, and `SPLASH_FLASH_SAVED_OPERANDS_RESIDENT=1` to
the previously qualified v4 routes. Local serving applies this profile only
after its exact source/model/Apple M5 Ultra hardware gate passes. It probes
hardware once and reuses the tuple for optional expert-store selection.

The normal v5 launcher qualified saved dense operands, top64 INT8 experts, and
residency together at 98.687/151.965 tokens/s for short single/four-request
workloads and 65.206/79.421 tokens/s for 2K single/four-request workloads. The
fresh oMLX reference was 89.710/112.084 and 61.995/77.993. Each cohort has two
measured samples, greedy generation, 128 output tokens per request, and zero
prompt-cache reuse. These are matched HTTP measurements; CPU helper checks do
not establish inference speed. The complete result is saved in
`build/release/flash/saved-format-default-v5-summary.json`.

Saved paths are dynamic optional defaults and are omitted from the accepted
static profile. `_qualified_saved_operand_defaults` resolves
`ROOT/install/local-models/Flash-Next-operands-v1` only after checking exact
source and aligned manifest bytes, the qualified OnePlusWeight fingerprint,
saved manifest bytes, and both manifest checksum files. Its manifest SHA256 is
`433e8a0ea5150fc063b7ccd02fc91191ba08032640ec2fd5d63cff5ece129512`.
The native loader validates source projection geometry and hashes selected
padded payloads before readonly zero-copy Metal mapping.

`_qualified_saved_int8_expert_defaults` selects
`ROOT/install/local-models/Flash-Next-int8-experts-top64-v1`, with manifest SHA256
`12593570ee67b62ddeadb951238879b780368cf99e096aa38a7d7771b9dc5c29`
and plan SHA256
`d0b1f58c87eb4292ea6e7d04eec55f0c722ee7583468dd58e45c2fa3f476d02d`.
It reuses the source/layout witness, checks all 48 layers and 64 selected experts
per layer plus gate/up/down dimensions/dtypes, and requires Apple M5 Ultra with
at least 256 GiB of integer physical RAM and effective BLOCKED_MOE=`1`.
Top128 remains available through an explicit manual path; missing top64 never
falls back to top128 implicitly. INT8 experts are a declared numerical
alternative to the original Q4 expert coefficients.

An absent artifact retains ordinary cache conversion or Q4 expert routes. A
present default artifact with corrupt or unqualified metadata raises an error;
payload corruption is rejected by native verification. Launcher helpers never
scan multi-GB payloads. Caller dictionaries and the process environment are
copied before defaults are merged.

Explicit caller values remain authoritative. Exact
`SPLASH_FLASH_OPERAND_STORE=0` and `SPLASH_FLASH_INT8_EXPERT_STORE=0` opt out
before source or artifact inspection; empty values remain invalid. Explicit
SAVED_OPERANDS_RESIDENT=`0` disables its implied residency request. Disabling
Q4X8 or BLOCKED_MOE clears an implied DIRECT_A flag; disabling QSA_MPP or
QSA_F32 clears an implied QSA_ROW_TILES flag. BLOCKED_MOE=`0` removes an implied
expert-store path instead of assigning filesystem path `0`. Explicit child
flags or paths are preserved.

The static/effective review snapshots remain under
`build/release/flash/local-profile-v5-candidate{,-static}.json`.
The isolated native opt-out checker is
`build/flash-operand-optout-cpu/flash-operand-export`; its empty-weight self-test
constructs no Metal device and proves both exact-zero selectors return before
source or payload inspection. Active integration tests cover qualified normal
serving, one hardware probe, absent artifacts, caller values, and malformed
metadata. Checkpoint payloads and remote model roster remain unchanged.
