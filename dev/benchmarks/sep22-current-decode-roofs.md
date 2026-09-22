# Current decode measurements and bandwidth scenarios

All rates are aggregate tokens/s. Observations use actual emitted native tokens after the first emission. The bandwidth scenarios use 1173.09 GB/s of measured effective resident-array read payload; physical DRAM traffic was not measured.

| Native lanes | Standard optimistic ceiling | Standard streaming reference | MTP optimistic ceiling | MTP streaming reference | MTP observed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 197.72 | 197.72 | 195.98 | 116.88 | 53.23 |
| 2 | 275.04 | 209.65 | 296.26 | 182.20 | 104.31 |
| 4 | 309.71 | 213.09 | 388.44 | 201.49 | 157.29 |

The singleton observation is the qualified Q4 rowpair path: 4002.16 tok/s median uncached prefill, one of three trials below 4K, original22 task comparison with no new regressions. The two- and four-lane observations use the integer verifier with older prefill; restored prefill composition and original22 batch qualification remain pending. Separate restored standard observations are 60.93 and 93.27 tok/s for two and four lanes. No current qualified standard singleton score is substituted from an older numerical policy.

Ceiling = effective payload bandwidth × useful tokens per cycle / modeled operand and state bytes. It omits arithmetic, dispatch, transaction amplification, host work and imperfect cache sharing. MTP singleton uses the current 150 accepted drafts over 105 cycles, or 2.42857 useful tokens per cycle. Batch inventory and acceptance assumptions come from the source-backed earlier report. Perfect four-token acceptance would raise the optimistic scenarios to 322.79 / 367.13 / 475.27 tok/s, but that acceptance is unobserved.

The current singleton verifier costs 35.57 ms of its equivalent 45.63 ms cycle. Holding acceptance and other costs fixed, a measured 1.5× or 2× verifier improvement would yield 71.91 or 87.23 tok/s. These are conditional sensitivities, not achievable speeds established by this analysis. Native common graphs support at most four lanes.

Exact input report hashes and full-precision calculations: [machine-readable evidence](/Users/mweinbach/Projects/splash/build/release/flash/sep22-current-decode-roofs-and-observations-v1.json). Reproduction: [CPU-only script](/Users/mweinbach/Projects/splash/dev/benchmarks/sep22_decode_roofs_current.py).
