The current stock Full512 executable's HC-up F32 path requires physical rows4
through16. Enabling `SPLASH_FLASH_HC_UP_F32_MPP=1` alone does not select it for
singleton AR rows1. Actual warmed standard singleton counters recorded24,735
attempts, zero eligible/cache calls, and24,735 unsupported-geometry calls. The
batched AR executor directly selects raw fused HC-up, including batch4. Target
verifiers select the F32 bridge; their other dense F32 policies are enabled by
the actual matrix's sharedDenseRoutes/FLOAT_DENSE_CACHE settings.

The exact97 target HC-up matrices have202,956,800 original packed-code/scale/
bias bytes. Their F32 operands occupy1,271,398,400 unique bytes, an expansion
of1,068,441,600 bytes. This expansion is part of the four-row verifier inventory,
and is absent from singleton AR. A hypothetical forced-F32 singleton would
lower its optimistic traffic ceiling from197.72 to167.55 tok/s; the current
stock implementation does not take that route.

Root's standalone array benchmark measured1173.0877 GB/s for rotating8 GiB
reads and1174.2670 GB/s for alternating16 GiB source-pair reads. The lower read
median calibrates these scenarios. Copy read+write measured1079.6094 GB/s,
which is retained separately. All seven samples and every CTA/source/destination
validation passed. These are sustained effective payload rates; no physical
DRAM transaction counters were collected.

Regenerate the CPU-only current-route report:

```sh
.venv/bin/python dev/benchmarks/decode_target_sep21/current_routes.py \
  --out build/release/flash/sep21-decode-current-route-roofs-v2.json
```

The script reads model JSON metadata, the measured binary's source snapshot,
the actual standard/MTP matrix, and the memory benchmark. It checks binary
identity, source route guards and the resolved cache/fusion flags, and retains
provenance hashes. No model payload is opened and no GPU/device work executes.

Two different models address cache sharing. The optimistic unique-operand
footprint assumes that every source dense/vocabulary matrix and unique expert
is fetched only once for a native cohort. The streaming reference instead
repeats independently requested raw dense rows and gathered expert routes,
while preserving explicit F32/vocabulary matrix reuse across MPP row tiles.
Actual hardware caches can serve some repeated requests. Neither scenario
measures physical traffic or proves attainable decode speed.

| Native cohort | Standard unique / streaming reference | MTP current prefix unique / streaming reference | Measured standard / MTP |
| --- | ---: | ---: | ---: |
|1|197.7 /197.7|257.2 /152.9|35.59 /68.36|
|2|275.0 /209.6|296.3 /182.2|60.65 /99.28|
|4|309.7 /213.1|388.4 /201.5|93.39 /139.25|

Rates are aggregate tokens/s. Current MTP prefixes are3.1875,3.22785 and3.26923
committed tokens per lane per cohort cycle, derived directly from accepted
prefixes and verification-cycle counts. Perfect four-token acceptance increases
the numerator but also increases streamed true-feature head-fold work; the JSON
records that optimistic scenario separately.

Current verifiers select118 /202 /296 F32 matrices at physical rows4 /8 /16.
Their unique footprints are3.248 /9.068 /12.098 GB, replacing0.619 /1.521 /2.054
GB of packed operands. At rows16, HC-up, QSA N32 output and some dense kernels
use two M8 row tiles. Their explicit requested F32 matrix footprint is22.345 GB,
which the12.098 GB unique footprint alone hides. Standard batch4 excludes all
97 HC-up F32 matrices and has1.561 GB of selected-cache expansion.

The trained MTP head retains original packed operands. Streaming fold work uses
the actual true committed prefix, plus two subsequent draft calls. Its trained
fc_hidden projection applies the original matrix independently to four streams;
vocabulary MPP shares its source operand across native cohort lanes.

For independent scalar lanes executed serially with no guaranteed cross-request
cache reuse, aggregate memory-scenario rates remain the singleton values as
batch grows; they do not multiply by batch. Native batch8/16 common graphs are
unsupported. The report models those requests as serial four-lane cohorts with
the batch4 aggregate scenarios, explicitly labeling the unsupported enlarged
graph. Observed batch4 rates are cohort measurements, not batch8/16 evidence.

More defensible engineering targets preserve actual whole-wave decode costs
and acceptance. The actual native verifier GPU medians were37.09 /51.72 /73.97
ms per cohort cycle. A twofold verifier speedup, with all other observed costs
retained, conditionally predicts113.5 /164.8 /229.7 aggregate tok/s. A1.5-fold
improvement predicts93.0 /135.1 /188.8. These require actual kernel gains and
subsequent whole-worker verification; they are not achieved or promised speeds.

Scratch/padding, gathered activation reloads, SSD preparation, partial recurrence
replay, arithmetic, dispatch, host work and transaction amplification are omitted
from the traffic scenarios. Expert sharing uses independent uniform lane unions
and one older diagnostic temporal union, which can differ from the current
numerical derivative's routing. The measured kernel targets therefore take
precedence over an optimistic memory roof when defining the next experiment.
