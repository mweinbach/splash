The HC-down control computes one full10240-term F32 dot per SIMD group. It
does not compute four independent stream dots or a BF16 stream sum. Splitting
that dot into stream partials would change its accumulation tree. This private
candidate instead changes dispatch grouping: one32-thread CTA per output,
against the control's128-thread CTA containing four independent output SIMDs.

The timed wrapper validates SG1, then directly calls the unchanged original
SG4 helper with output-index remapping. Lane32 K order, group/block order,
coefficient reconstruction, `simd_sum`, BF16 raw projection, divide-by4,
compiled BF16 SiLU and precise BF16 injection gates remain the same body.
It supports original Q4/Q5/Q6/Q8 and G32/G64/G128 formats, including independent
mixed-format injection matrices. No cache is allocated, weights are borrowed
from their original immutable owner, and no model graph/default is selected.

```sh
.venv/bin/python dev/benchmarks/hc_down_sg1_sep21/prepare.py \
  --build build/hc-down-sg1-sep21-v4
```

Strict host/shader compilation and CPU self-tests create no GPU backend or
model payload read. The prepared source/object/AIR closure has247sources and
82copied inputs. Original helper-body comparisons,160-byte parameter ABI,
200-byte command timing ABI and output remapping checks pass. Earlier failed
host builds remain preserved; use the completedv4.

Root invokes GPU mode only after serializing other GPU/model work:

```sh
SPLASH_FLASH_ALLROWS_FULL512_TARGET=1 SPLASH_FLASH_PLE_SSD_STREAMING=1 \
SPLASH_FLASH_INT8_EXPERT_STORE=CERTIFIED_FULL512_DIRECTORY \
FLASH_HC_DOWN_SG1_ROWS=1,4 FLASH_HC_DOWN_SG1_PAIRS=10 \
build/hc-down-sg1-sep21-v4/oracle --gpu \
  build/hc-down-sg1-sep21-v4/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 NEW_REPORT_JSON
```

The default six source roles cover Q4/Q5/Q6/Q8, mixed Q4-down/Q5-injection and
the final mixer without injection. Optional `FLASH_HC_DOWN_SG1_PREFIXES` admits
a comma-separated subset; `FLASH_HC_DOWN_SG1_INPUT` accepts an exact-sized
normalized BF16[rows,10240] fixture. Otherwise the report labels its deterministic
synthetic inputs. Row selectors admit only1/4. Only Root mode loads source
weights and creates a Metal device.

Untimed SG4/SG1 probes expose full rawF32 and BF16 boundaries. Literal witnesses
must reproduce shipping control outputs, and candidate probe/timed outputs
must match. Every rawF32/BF16, activation and injection word must be identical.
Candidate-own BF16 raw values also run through an independent compiled epilog
with the original scalar operations. Canary, sticky diagnostic and immutable
source/input checks gate timing. Raw taps use compact320/324 stride, including
multirow injection-absent fixtures.

Each route receives at least150ms actual GPU work after all proof reads. Ten
even AB/BA timing pairs preserve first/second-position balance. No CPU tensor,
diagnostic or guard reads occur from warmup through the final timed submission;
timer metadata is the only returned timing data. Reports retain position strata.
Component exactness or speed does not establish whole-worker decode gain,
cache/state continuity or service quality; Root owns later qualification.
