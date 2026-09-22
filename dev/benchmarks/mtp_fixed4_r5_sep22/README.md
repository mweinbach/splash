Private fixed4/R5 QA and canonical driver source. No inference has run through
this adapter. Existing original22 bodies, graders, budgets, context and MTP
eligibility remain unchanged. Actual saved policy remains cap4; only the old
cap3 gate receives a private copied singleton-cap view. The original comparator
has one reversible edit to compare registered cap3/cap4 policy summaries. Every
other summary key and the original reconstruction/grade code remains intact.

`prepare.py --output-dir NEW_BUILD_DIR` writes source plus inverse journal only.
Bound command preparation additionally requires `--binding PATH
--binding-sha256 EXACT_EXTERNALLY_REGISTERED_SHA` after Root whole-state proof.
No fabricated or placeholder binding is provided.

Binding schema: `Root-registered-fixed4-R5-exact-runtime-binding-v2`.
Required fields are `schema`, absolute `worker`, exact `source_identity_sha256`,
`exe_sha256`, `library_sha256`, `metadata_sha256` for all four files below,
positive exact `native_state_counts`, `native_oracle_sha256`, `native_report_sha256`,
`native_allocation_guards` and `native_source_scope="MAIN ONLY"`. Extra binding
keys are rejected. The allocation guards must contain exactly the six actual
source-defined guard names below, each with Boolean true.

Fixed files: `CPU_READY.json`, `compiled-cpu-seal.json`,
`overlay-manifest.json`, `Root-r5-native-qualified.json`, `splash-flash`,
`splash.metallib`, `R5-qualified.air`. The leaf AIR must equal the registered
50002976851cd0bf2cf0f133c0164ccf177dfc1e7493b28563ba8fdbd1bb7ac3.

Compiled metadata schema:
`singleton-R5-integer-current-Q4-worker-compiled-source-v1`. Required exact
fields: pass, source/exe/library/planner AIR SHA, parent executable663663 and
library754028 SHA, public_headers_changed=false, new_GPU_allocation_bytes=0,
GPU_work=false, whole_state_qualified=false and only Forward/Worker changed.

Native receipt schema: `singleton-R5-current-Q4-fixed4-native-state-proof-v1`.
Require pass/qualification_complete/Root_GPU_executed/backend_destroyed true,
current source/exe/library/planner SHA, maximum_verify_rows5, exact positive
state_counts, exact oracle/root_report SHA, original22_qualified=false,
native_source_scope="MAIN ONLY", teacher_head_proved=false and exact
allocation_guards: target_ledger_matches_workspace, target_fits_category_plan,
workspace_plus_state_plan_fits_reservation, state_fits_state_plan,
workspace_plus_actual_state_fits_reservation, live_backend_delta_fits_reservation.

Require allocation_owner_ledger with after_target_equals_mapped=true,
after_model_destruction_bytes=0, governor_reserved_bytes=0,
governor_denied_reservations=0, host_measurement_valid=true and
backend_stopped=true. These follow actual source ledger/teardown/governor checks.
Require memory_admission_scope="explicit_governor_reservation_and_six_ledger_guards".
The native oracle makes explicit governor reservations before target/state
growth and checks aggregate owned/category ledgers. It does not install a
per-physical-allocation admission callback.
The native oracle did not sample physical device current or peak allocations:
device_allocation_measurements_present must be Boolean false. Legacy
allocation_axes and physical device zero/peak claims are rejected. This receipt
is separate from future original22 tasks and performance. Admission reads only
the Root receipt, not the native data report.

Actual H3 cycles retain original raw26/HC97/preflight48/compact48 coverage.
The new R5 section requires exactly48 full-layer calls per actual H4 cycle and
five rows per call. EOS/quota-shortened cycles use the existing branches.
