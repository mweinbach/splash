# Exact R4 HC producer padding component

Root command after sole GPU ownership:

```sh
build/hc-pad-producer-sep22-v6/run-root.sh
```

This isolated component retains the existing SG4 HC-down helper and the original F32 M8N32SG4 up entry. Row-zero/lane-zero output owners additionally write literal positive zero to inactive activation rows 4–7. The consumer still receives BF16[8,320], identical extents/strides, coefficients, operator descriptor and arithmetic. The candidate removes one padding dispatch from each of 97 target HC calls: baseline 291 dispatches, candidate 194. The initial source-backed saving estimate is about 0.47 ms of diagnostic pad time, not an achieved gain or a twofold speedup.

The private API requires explicit Verify scope, physical R4, and an exact 5,120-byte eight-row activation view. Prefill, ordinary AR, trained head and batch scopes are rejected. No production overlay is installed. The existing target scratch admits the view at maximumRows2048; the component uses matched guarded fixtures. No candidate coefficient/cache/workspace allocation is added relative to its control.

All 97 actual target HC source prefixes are covered, using original packed down/injection operands and exactly 97 saved original F32 up matrices (1,271,398,400 payload bytes). Normalized inputs are declared deterministic fixtures by default. Root may separately supply a directory through `HC_PAD_INPUT_DIRECTORY` when preparing a new reviewed command; every `<canonical-prefix>.bf16` file must contain exactly BF16[4,10240]. The isolated 97-call graph is not the whole target verifier or a service-quality proof.

Untimed probes expose all active raw down F32/BF16, post-Silu activation and injection gates, plus all 40,960 raw up F32/BF16 entries and 10,240 BF16 mix entries per role. Both routes must match exactly, including the complete eight-row padded tensor. Probe results also match their unchanged production consumer outputs. The helper/probe source journals restore original bodies after only tap/name/pointer changes; no dot or sigmoid arithmetic is replaced.

The private 176-byte parameter structure embeds canonical 160-byte down parameters unchanged. Canonical up parameters remain 32 bytes, and timing ABI is 200. All 36 private down exports require the full {80/81,4,1} group grid and 128 threads before writes, rejecting partial and excess dispatches. Root execution includes 16 shader no-write metadata/partial-geometry checks, seven host alias/scope checks and two nonfinite sticky checks. Canary, sticky, immutable input/coefficient and full output checks precede sustained warmup and repeat after all timing.

Each timed route receives at least 150 ms of actual GPU warmup, followed by ten balanced AB/BA pairs (five ABBA blocks, ten samples per route and five per timing position). Shared tensor/diagnostic/canary reads stop before warmup and resume only after every timed position completes. Timed UP is the original immutable AIR entry; the raw-F32 probe is not timed.

The final build seals 287 source files, the current teacher-v5 parent's 53 effective nonworker objects, four original immutable AIR files and the private candidate AIR. The component library contains the HC, padding and cache-expansion functions required by this diagnostic. It intentionally does not claim identity with the complete worker library. Oracle SHA is `4feeaeccd09874df3993cdf7b0772d41327af66344d81b59addfadd339002ab5`, library SHA `1f44e10a9653c5233a097f17705dbeffdd9350ac4dd6cb27e5749e4dfa0c0f16`, and Root command SHA `d8fefe669a401a019e7605dd4cc819ca2d850e7496148605d4bc2e07dd21d585`.

Metal compilation, C++ ABI/host compilation and CPU eligibility/ownership/inventory checks pass. Independent source/host review found no fatal issue. Actual numerical and timing qualification remains Root-owned; agents performed no GPU calls, model/input tensor reads or hashes, new-agent spawning, or production edits.
