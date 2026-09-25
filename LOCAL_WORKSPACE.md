# Local inference workspace

The local default, and the only pinned profile, is `m5-ultra-flash-next-v18`:
the megakernel worker from
[dev/benchmarks/flash_opt_sep22](dev/benchmarks/flash_opt_sep22/README.md),
pinned by SHA-256 at `build/flash-opt-sep22-v18/`. It runs the original Q4
experts with tiled few-row decode, batched-verification and prefill kernels:

| 2,048 input / 256 output, greedy, MTP depth 4 | v18 |
| --- | ---: |
| Single-stream decode, ten fresh prompts | 169.8 tok/s (19.6 ms cycle) |
| Single-stream decode, canonical coding prompt | 180.5 tok/s |
| Prefill, canonical prompt | about 4,200 tok/s |
| Two / four concurrent lanes, aggregate decode | 161.6 / 196.8 tok/s |
| 49 / 396-token fresh prompt, prefill wall time | 78 / 241 ms |

The frozen semantic suite passes 20/22 (the same two arithmetic cases fail in
every build). For reference, the qualified September 22 baseline this worker
started from measured 4,006.9 tok/s prefill and 61.54 tok/s decode on the same
canonical run.

The profile keeps that baseline's measured flag envelope and adds the
`SPLASH_OPT_*`/`SPLASH_MK_*` megakernel flags. It still requires the qualified
Full512 expert store and saved operands, streams PLE rows from SSD, and
defaults to 16K context (an explicit context setting overrides it). With the
Q4 experts selected, the INT8 store is not mapped; the Q4 experts and a 70 GiB
tile copy stay resident instead. Original model payloads and oMLX preferences
are unchanged. The earlier pinned profiles (v13-v17) and the v14-v17 runtimes
were removed; their measurements remain in the benchmark README.

The September 21 update keeps MTP draft depth3 and the exact Q5/Q6 QSA output
projection tile (`SPLASH_FLASH_QSA_OUT_F32_N32=1`), and enables exact teacher
cache priming plus the authenticated2K cached projection tiles. Matched
uncached2K/256 HTTP prefill improved2,097→2,424 tok/s with identical outputs. Exact public bulk QSA/SG8 now
adds fresh2K main-target scheduling. Public qualification observed2,539 tok/s;
its historical +4.74% comparison had a different idle-maintenance policy. Dense traversal is available
as an opt-in (`SPLASH_FLASH_DENSE_TRAVERSAL=1`); matched HTTP results did not
justify enabling it by default. See [locality experiments](dev/benchmarks/ultra-locality-results.md).

Start the current configuration:

```sh
./splash serve --model local/Qwen3.8-Flash-Next-oQ4e-mtp \
  --local-package install/local-models/Flash-Next-oQ4e-mtp-v1 \
  --port 8000 --no-webui
```

Use `--no-ple-ssd-streaming` to keep the n-gram table resident instead.
Use `--ple-ssd-cache-mb 128` to change its bounded row cache.
Idle residency maintenance is disabled in this measured configuration.

Active code and evidence:

| Purpose | Location |
| --- | --- |
| Launcher and accepted profile | `install/launcher.py`, `.splash-local-profile.json` |
| Selected runtime and matching library | `build/flash-opt-sep22-v18/splash-flash`, built from `dev/benchmarks/flash_opt_sep22/worker` |
| Flash execution, weights and SSD store | `runtime/flash/` |
| GPU kernels and ABIs | `runtime/metal/kernels/shared/`, `runtime/metal/abi/` |
| Model and qualified saved operands | `install/local-models/` |
| Historical public defaults and opt-outs | [v12 profile](dev/benchmarks/flash-local-profile-v12.md) |
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
