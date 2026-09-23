# Local inference workspace

The current local default is `m5-ultra-flash-next-v16`: the v13 environment and
model artifacts with the optimized worker from
[dev/benchmarks/flash_opt_sep22](dev/benchmarks/flash_opt_sep22/README.md)
(pinned at `build/flash-opt-sep22-v16/`). Single-stream decode runs at a 28.3 ms
MTP cycle (v14: 32.7 ms, v13: 64 ms), about 116-119 tok/s on ten fresh prompts
(v14: 102.5, v13: 51.5). Canonical 2,048/256 prefill is 4,208 tok/s (v14: 4,097,
v13: 4,014), short fresh prompts prefill up to 16% faster than v15, and batched
decode reaches 117.1 aggregate tok/s at two lanes and 143.9 at four (v14: 99.3
and 121.2). The frozen semantic suite passes 20/22, the same as v13. To return
to an earlier runtime, write `LOCAL_PROFILE_V15`, `LOCAL_PROFILE_V14` or
`LOCAL_PROFILE_V13` from `install/launcher.py` to `.splash-local-profile.json`.

V14 added multi-row quantized GEMV/HC kernels for verification and drafts,
parallel PLE SSD reads, a reduced draft vocabulary and a batched-GDN fix; v15
the host-side and kernel work, v16 the 8-bit repack, short-prefill route and
fused shared-expert SwiGLU described in the benchmark README.

The qualified baseline, `m5-ultra-flash-next-v13`, measured **4,006.9 tok/s uncached
prefill and 61.54 tok/s native MTP decode** at 2,048 input / 256 output tokens,
greedy sampling, one warmup and three trials, with a 16K context limit.

V13 selects the pinned existing fixed-depth-4 worker and matching Metal library,
Full512 expert store, saved operands, SSD PLE streaming, and separate prefill and
verification routes. It defaults to 16K context; an explicit context setting
overrides that default. Original model payloads and oMLX preferences are unchanged.
V14-v16 keep all of these settings.

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
| Selected runtime and matching library | `build/flash-opt-sep22-v16/splash-flash`, built from `dev/benchmarks/flash_opt_sep22/worker` |
| Previous (v13) runtime | `build/R5-integer-currentQ4-fixed4-sep22-worker-v2/splash-flash`, matching `splash.metallib` |
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
