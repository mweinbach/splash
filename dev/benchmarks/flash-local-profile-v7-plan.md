# Local v7 promotion plan history

This CPU plan preceded Root's authorization. Active defaults are now v7;
see `flash-local-profile-v7.md`. The proposed change from v6's 34 defaults added
`SPLASH_FLASH_HC_UP_F32_MPP=1` and `SPLASH_FLASH_GDN_LAZY_ROLLBACK=1`, for 36.
Root qualified the combined service and same-build performance, then authorized
local promotion.

HC-up's operative parents are FUSE_HC and FLOAT_DENSE_CACHE. It uses the existing
original-F32 coefficient cache/workspace for R4..16 and the10240×320 HC-up
matrices. DENSE_CACHE, QMV_F32 and MTP are not prerequisites. Lazy rollback's
only mandatory parent is FUSE_GDN; GDN_STAGED, batching and MTP are unrelated
to its public verification helper. Explicit parent0 should clear only an implied
child, while explicit child values stay authoritative.

Promotion touches `install/launcher.py`, `.splash-local-profile.json`,
`dev/tests/engine/test_local_profile.py`, `dev/tests/engine/test_saved_operand_profile.py`,
and active-profile documentation. Existing source/layout/norm/hardware guards,
optional dense/TOP64 pins,2048-row windows, store0 opt-outs, absent fallback,
corruption rejection, one hardware probe, and manualTOP128 must be preserved.
Historical v5/v6 review helpers must retain30/34 flags when the active profile
becomes v7. Original-text residency and private loader/indirect modes stay off.

Build integration is already present: HCFused is in the existing Flash source
list, GDNLazyRollback is explicitly listed, and both new shaders are included
by the shared-kernel wildcard. HC uses the existing32-byte
FlashFloatDenseSmallRowsParams. Lazy verify/replay/copy use56/72/8-byte ABIs with
host/shader static assertions. Production ComputeDispatch/backend ABI is
unchanged. BatchVerifyGDN helper signature changed, so use a fresh complete host
rebuild and its matching metallib rather than mixing v6 objects.

The runtime flags do not change compiler configuration; dependency files,
source build identity and kernel input-name digests track rebuilt implementation.
Singleton and joint memory planners already reserve lazy arenas before
construction. Root should confirm actual workspace fits both planners and exact
retained-prefix/terminal/cancellation behavior during combined qualification.

The JSON plan in `build/release/flash/local-profile-v7-promotion-plan.json`
records source hashes, exact dependencies, file list and required CPU assertions.
No profile, source or test files were changed by this plan. No GPU ran.
