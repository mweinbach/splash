The open hypothesis is changing cached F32 coefficient B to HALF while retaining the current BF16 input, whole-K M8/N32 or N64/SG4 geometry and BF16 store. This is an explicit numerical alternative. The installed SDK supports BF16 × HALF → F32; that fact does not establish faster execution.

Profile the real current fixed4 target before building. Its 39.84 ms verifier budget is a mixed-depth average: only25of79 cycles proposeddepth4. Older ordinaryVerify4 family timings do not establish the current budget. A20% native-rate gain at unchanged acceptance needs8.741 ms saved from the52.448 ms equivalent cycle; a2× family must cost at least17.483 ms.

If attribution warrants a primitive, begin with one QSA output K6144/N2560 at physicalR5:62.91 MB F32 control versus31.46 MB HALF, keeping80 CTAs withN32. Root owns coefficient and actual-proposal input capture. Require conversion census, original-coefficient FP64/error and unchanged numeric/guard gates before balanced read-free timing. Overflow and nonfinite conversion reject the alternative. No full-model HALF sidecar or integration is authorized.

This differs from closed HALF prefill tests, which converted BF16 coefficients/input and used M128/M256. Full state/future and original22 model quality remain necessary after any component win.
