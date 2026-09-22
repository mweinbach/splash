# Deferred SIMD32 two-column vector variant

After the sealed SIMD32/O4 and SIMD16/O8 screens, a possible next candidate
assigns two output columns to each SIMD32 group, giving eight outputs per
128-thread CTA. Each group would reuse one adjacent BF16 vector across two
`char4` coefficient rows, with four independent accumulators per column.
The original late-scale and BF16 SwiGLU boundaries would stay unchanged.

Compared with SIMD16/O8, this would avoid four activation shuffle instructions
per K chunk and halve the chunk-loop trip count, while still producing eight
output rows per CTA. The SIMD32 certificate depths would remain28 additions for
K2560 and13 for K640. It would need its own dense tap and registered stage,
sampled-F64, scalar, malformed and warmed shipping-control checks.

This is an unimplemented hypothesis. Historical scalar C2's three cold R2
samples used a different baseline and cannot predict a gain against the current
gathered SG4 shipping control. Root requested actual warmed SIMD32/O4 and
SIMD16/O8 results before preparing this variant. No further source or CPU
compilation has been performed for it.
