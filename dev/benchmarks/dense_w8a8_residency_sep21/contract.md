The execution-safety conclusion is that ordinary direct legacy bindings retain
dispatch-time residency even when their allocations are absent from an attached
persistent residency set. Apple's documentation describes sets and encoder
declarations as complementary mechanisms, with declaration exemptions limited
to allocations contained in the set. This is a supported inference from those
contracts, rather than a literal claim that an attached set disables nothing.

Primary references:

- https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets
- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setbuffer(_:offset:index:)
- https://developer.apple.com/videos/play/tech-talks/10580/
- https://developer.apple.com/videos/play/wwdc2025/205/ (all-resources-in-sets requirement belongs to new Metal4 command path)

Actual backend is ordinary MTLCommandQueue at runtime/metal/MetalBackend.mm:1206,
ordinary commandBuffer at2265, computeCommandEncoder at2338, every direct
binding via setBuffer at2289. Separate embedded resources use useResources
at2284. Metal4 queue at1301 is separate sparse mapping only. Strong command
allocation retention at2230 proves lifetime separately from residency.

Inherited frozen core object SHA256:
f22b2da0996210e43d264feccd677ba5de89ac3e5eee0ccabed230ff5505dbd0.
Its dependency metadata names runtime/metal/MetalBackend.mm; archived source
SHA256 b7860f6f2eef91184caa7ef3c7dda6f66d9ec62b05b5e94533785cf004fba1d2.
CPU nm confirms legacy queue/commandBuffer/encoder/setBuffer/useResources and
residency-set calls. Frozen FlashDenseCache.cpp:414-416 binds weight.buffer
directly; tensors at179/227 strongly own original backing independently.

FlashOperandStore::mapTensor:416-429 creates one wrapSharedMemory allocation per
saved operand, with entry.allocated full extent. Every selected source extent
is already16K aligned. The policy requires exact dtype/shape/full-view saved
membership and84-owner/3,774,873,600-byte census before filtering a copied list.
No tensor, saved owner, mathematical fallback or governor allocation is deleted.

This contract does not prove idle pinning, OS reclaim timing, throughput gains,
startup admission, or model quality. Root's serialized pressure/performance and
fallback-path GPU qualification remain authoritative.
