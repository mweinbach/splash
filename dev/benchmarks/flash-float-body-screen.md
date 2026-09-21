The F32 body cache needs selection against the active quantized decode route. An earlier primitive comparison used the generic affine control and overstated gains. Screening commands here explicitly enable `SPLASH_FLASH_QMV_F32=1`; the oracle records its actual raw pipelines and the SHA256 of the immutable Metal library it loaded.

The corrected source checks in `build/release/flash/float-body-boundary-v3-corrected.json` establish fidelity for the layer-0 HC down and GDN output projections at four rows. They do not establish a performance gain: the F32 cache runs at **0.506×** and **0.384×** the active quantized control, respectively. These two routes should remain unselected at four rows.

CPU source inventory finds **508** eligible body matrices and **41** distinct role, shape, bit-width, and group-size combinations. Layer 0 uses GDN, layer 3 uses QSA, and layer 1 provides the first PLE projections. The quick screen covers 21 representative groups; the expanded screen covers all 41. BF16 router weights `[512, 2560]` belong to a separate BF16 matrix screen. GDN A/B and HC block injection are also excluded because their output widths are not divisible by 64.

Prepare commands without creating a Metal device:

```sh
python3 dev/benchmarks/flash_float_body_screen_plan.py \
  --package install/local-models/Flash-Next-oQ4e-mtp-v1 \
  --library build/flash-float-dense-cache/body-screen-v4/splash.metallib \
  --oracle build/flash-float-dense-cache/body-screen-v4/flash-float-dense-cache-oracle \
  --output-dir build/release/flash/float-body-screen-v4
```

Root runs one GPU job at a time:

```sh
sh build/release/flash/float-body-screen-v4/run-float-body-quick-screen.sh
# Expanded source coverage, after reviewing the quick screen:
sh build/release/flash/float-body-screen-v4/run-float-body-expanded-screen.sh
```

Each source matrix is tested at 4, 8, and 16 rows, with M8N64 and M16N64, warmup and six alternating timing samples per route. Caches are resident one source matrix at a time. Timing includes row-padding dispatches. The numerical checks retain bit-exact BF16 inputs, finite outputs, sticky diagnostics, row guards, workspace canaries, sampled independent F32 dot references, and original F32 coefficient samples.

The head retains the strict relative L2 limit of `1e-4`. A body matrix failing that original limit may receive a separate, explicit BF16 boundary proof. The proof verifies every original F32 coefficient used, computes every output column independently in double, and permits only exact or adjacent BF16 cells. An adjacent cell must share a midpoint within the deterministic F32 dot error bound, and the cached/raw outputs must differ by at most one BF16 ULP. This demonstrates compatibility with a reordered F32 reduction; it does not claim identical hidden accumulators or model generations. The bound uses the deterministic `gamma_K` analysis, not assumptions about random rounding errors. [Higham and Mary, 2019](https://eprints.maths.manchester.ac.uk/2731/1/paper.pdf)

Screening continues a numerical failure to collect the other cases. It preserves `accuracy_pass: false`, top-level `pass: false`, and exit status 1. A failing matrix remains ineligible. The report distinguishes the original strict acceptance from a boundary exception. Full-column double references are computed only when needed to justify an exception.

The plan preparer and CPU tests execute no GPU work. Root completed both source screens: all 126 quick cases and all 246 expanded cases passed numerical qualification against the active quantized control. Root owns GPU scheduling and whole-model validation.

The resulting `flashFloatDenseSmallRowsPolicy` selects only measured role, shape, bit-width, and group-size combinations. Each chosen tile needs at least a 15% GPU gain and no wall regression in every available quick/expanded screen. Selection uses median absolute cached GPU time across screens; speedup ratios alone can favor a tile whose raw timing varied. The policy semantic tag is `m5-ultra-source-qualified-f32cache-r4-r8-r16-padding-floors-v1`.

The formats matter. QSA output Q5/Q6/Q8 with group 64 wins strongly at four rows; Q4 output remains raw through 16 rows. Q4 QSA key/value/indexer projections also remain raw, while later-layer Q5/Q6/Q8 formats can win at larger row counts. HC down and shared gate/up remain raw for every tested format. Shared down selects the cache only at 16 rows. Unknown roles or changed geometry/quantization retain the raw fallback. Trained MTP body roles are not included in the main-body source proof; the shared vocabulary head keeps its separate qualified policy.

Body rows 2–3 remain raw. Rows 4–7 can use the qualified row-4 tile because its padded extent stays fixed. Row 8 uses the measured winner. Rows 9–15 select M16 only where the row-8 M16 case itself wins; using an M8 row-8 winner here would double the padded extent and could regress. Row 16 uses its measured winner. The caller still needs matched whole-model validation before enabling this selection as a default.
