# Local inference workspace

The current local default is `m5-ultra-flash-next-v10`: SSD streaming for the
PLE n-gram table, a native64 MiB row-cache budget, and the qualified faster
routes from v9. Other weights remain GPU-accessible in memory. The accepted
profile has39 static settings; saved operand and expert paths are qualified
at launch. Original model payloads and oMLX preferences are unchanged.

Start the current configuration:

```sh
./splash serve --model local/Qwen3.8-Flash-Next-oQ4e-mtp \
  --local-package install/local-models/Flash-Next-oQ4e-mtp-v1 \
  --port 8011 --max-context 8192 --no-webui
```

Use `--no-ple-ssd-streaming` to keep the n-gram table resident instead.
Use `--ple-ssd-cache-mb 128` to change its bounded row cache.
Idle residency maintenance preserves accessibility of the remaining weights;
`SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE=0` disables it before launch.

Active code and evidence:

| Purpose | Location |
| --- | --- |
| Launcher and accepted profile | `install/launcher.py`, `.splash-local-profile.json` |
| Native runtime and matching library | `build/flash-next/splash-flash`, `build/flash-next/splash.metallib` |
| Flash execution, weights and SSD store | `runtime/flash/` |
| GPU kernels and ABIs | `runtime/metal/kernels/shared/`, `runtime/metal/abi/` |
| Model and qualified saved operands | `install/local-models/` |
| Current defaults and opt-outs | [v10 profile](dev/benchmarks/flash-local-profile-v10.md) |
| SSD memory/performance qualification | [SSD streaming](dev/benchmarks/flash-ple-ssd-streaming.md) |
| Idle-latency implementation | [Idle residency](dev/benchmarks/flash-idle-residency-fix.md) |
| Retained reports and cleanup records | `build/release/flash/` |

Workspace cleanup removed9,886 obsolete compiled artifacts, extracted tensor
copies and large raw profiling records, reclaiming7.628 GiB, plus generated
Python cache debris. Source, source snapshots, small fixtures, model stores,
active/qualified runtimes, report summaries and raw-trace hash manifests remain.
The raw trace recordings/large exports were intentionally deleted; their exact
recovery is not promised. Older unrelated27B results and builds are preserved.

Cleanup records are `workspace-cleanup-plan-v13.json`,
`workspace-cleanup-result-v13.json`, and `workspace-python-cache-cleanup-v13.json`
in the retained evidence directory. The cleanup phase preserved local work.

This workspace is now published to the independent private
`https://github.com/mweinbach/splash` repository. Only its private `origin` is
configured; there is no upstream remote or GitHub fork parent. Scheduled
upstream catalog refresh is archived under `dev/archived-workflows/` and GitHub
Actions are disabled for this initial private copy. Licenses and historical
attribution remain intact. Model/build/benchmark payloads are not uploaded.
