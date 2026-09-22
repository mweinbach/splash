# Independent CPU source and closure review

Reviewed artifact: `build/gemv-r1-ab-qsa-sep21-worker-v1`.
Reviewer: `/root/decode_kernel/bf16_dependencies`.

The review found no fatal issue within the inherited ordinary AR1 call-site scope. No GPU execution, model payload access, or artifact changes were performed by the reviewer.

- All 288 source seals and 123 frozen inputs match the manifest.
- An independent `clang -MM` census over 48 translation units identifies exactly six consumers of the modified Store/Forward headers. All six are rebuilt; MTP translation units are not consumers.
- The link closure has 49 unique host objects plus four core objects, without stale or duplicate consumers. The shader closure retains 76 parent AIR files and adds the qualified vector AIR.
- Shipping arithmetic, certificate files, and source identity `bd384554f00dafdc285bf6df2dd34bebfe2d963fa1c955e0834c8a08e656f29b` retain the existing qualified R1 expert-I8 vector producer.
- The only Worker entry point for the vector route is the explicit ordinary pending-token `forwardDecode` call. Its wrapper requires one token and nonempty prior state; the route also requires gathered MPP, physical R1, and no verification.
- Prefill (including R1 suffixes), MTP seed/verification/head, R4, and batch paths retain their prior routes. The nine MTP source files are unchanged.
- AB/QSA source blocks and cache/workspace planning are retained. Static source identity excludes changing route counters.
- The CPU source witness and six strict-policy process modes pass. Each policy process performs approximately 49,000 checks, approximately 295,000 in aggregate. The original Worker CPU self-test also passes.

Scope detail: the inherited ordinary pending-token branch also handles an MTP request's final output-budget-at-most-one AR fallback. That terminal fallback may use the R1 vector; this is distinct from MTP seed and verification routing. A stricter exclusion of every MTP-owned request would require a new artifact that retains `forward()` for this fallback.

These are CPU source and closure results. Whole-model timing, outputs, and semantic qualification remain separate Root-owned evidence. The prepared commands and their hashes are bound by `build/gemv-r1-ab-qsa-sep21-worker-v1/ready-seal.json`.
