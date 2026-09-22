"""Private main-prefill W8A8 transforms; public headers/production stay intact."""
import re

PRIVATE = "dev/benchmarks/dense_w8a8_sep21"
CHANGED_FILES = {"runtime/flash/FlashForward.cpp", "runtime/flash/FlashWorker.mm", "dev/benchmarks/prefill4k_attribution.mm"}

def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f"Dense W8A8 frozen-source anchor drift: {old!r}")
    return text.replace(old,new)

def transform(relative, text):
    if relative not in CHANGED_FILES:
        return text
    text = f'#include "{PRIVATE}/worker_bridge.hpp"\n' + text
    if relative == "runtime/flash/FlashForward.cpp":
        text = replace(text, "  std::unique_ptr<FlashDenseCache> denseCache;", """  std::unique_ptr<FlashDenseCache> denseCache;
  const bool denseW8Requested = dense_w8a8_sep21::requested();
  std::unique_ptr<dense_w8a8_sep21::Cache> denseW8Cache;
  std::unique_ptr<dense_w8a8_sep21::Workspace> denseW8Workspace;""")
        text = replace(text, "    if (captureRoutes)\n      capturedRoutes = backend.allocateBuffer(", """    // Resource policy is fixed for both selector values. Source BF16 operands
    // are inspected/fitted only here, during authorized model startup.
    if (dense_w8a8_sep21::requiresCache(maximumRows)) {
      if (!denseCache) throw std::invalid_argument("private dense W8A8 worker requires DENSE_CACHE=1 at maximumRows>=2048");
      denseW8Cache = std::make_unique<dense_w8a8_sep21::Cache>(backend,weights,*denseCache);
      denseW8Workspace = std::make_unique<dense_w8a8_sep21::Workspace>(backend);
    }
    if (captureRoutes)
      capturedRoutes = backend.allocateBuffer(""")
        text = replace(text, "               const metal::MetalBuffer &diagnostics, uint32_t rows) {\n    if (int8Head", """               const metal::MetalBuffer &diagnostics, uint32_t rows,
               bool verification = false, bool singletonMain = false) {
    if (denseW8Requested && singletonMain && !verification && rows == 2048 &&
        denseW8Cache && denseW8Workspace && denseW8Cache->contains(prefix) &&
        dense_w8a8_sep21::addProjection(graph,*denseW8Cache,prefix,input,output,
            diagnostics,rows,verification,*denseW8Workspace,4)) return;
    if (int8Head""")
        # Only execute's singleton trunk projections enable this path. Existing
        # HC, vocabulary and batchProject call sites retain the default false.
        text = replace(text, "    impl_->project(graph, prefix, input, output, diag, rows);", "    impl_->project(graph, prefix, input, output, diag, rows, verification, true);")
        text = replace(text, "  return total;\n}\n\nstd::string FlashForward::kernelRoutes()", """  if (dense_w8a8_sep21::requiresCache(maximumRows))
    total += dense_w8a8_sep21::Workspace::plannedBytes();
  return total;
}

std::string FlashForward::kernelRoutes()""")
        text = replace(text, "  return std::string(flashAffineSemantics()) +", """  return std::string(flashAffineSemantics()) +
      std::string(dense_w8a8_sep21::selectionMarker(impl_->denseW8Requested)) +
      std::string(dense_w8a8_sep21::kCacheMarker) +
      (impl_->denseW8Cache ? impl_->denseW8Cache->identitySha256() : "none-maxrows-below2048") +""")
        text = replace(text, "  if (impl_->denseCache) append(impl_->denseCache->persistedWeightBuffers());", """  if (impl_->denseCache) append(impl_->denseCache->persistedWeightBuffers());
  if (impl_->denseW8Cache) append(impl_->denseW8Cache->immutableWeightBuffers());""")
        text = replace(text, "  if (impl_->denseCache)\n    for (const auto &buffer : impl_->denseCache->immutableWeightBuffers()) reject(buffer);", """  if (impl_->denseCache)
    for (const auto &buffer : impl_->denseCache->immutableWeightBuffers()) reject(buffer);
  if (impl_->denseW8Cache)
    for (const auto &buffer : impl_->denseW8Cache->immutableWeightBuffers()) reject(buffer);
  if (impl_->denseW8Workspace)
    for (const auto &buffer : {impl_->denseW8Workspace->codes,impl_->denseW8Workspace->inputScales,
        impl_->denseW8Workspace->dummyDot}) reject(buffer);""")
    if relative == "runtime/flash/FlashWorker.mm":
        text = replace(text, "      std::signal(SIGPIPE, SIG_IGN);", """      (void)dense_w8a8_sep21::requested(); // Freeze strict selector before paths, metadata or backend.
      std::signal(SIGPIPE, SIG_IGN);""")
        text = replace(text, "      plannedTrunk += FlashForward::expertCachePlannedBytes(weights);", """      if (dense_w8a8_sep21::requiresCache(prefillRows))
        plannedTrunk += dense_w8a8_sep21::Cache::plannedBytes();
      plannedTrunk += FlashForward::expertCachePlannedBytes(weights);""")
        if "savedResidencyLease.bufferCount() != 748" in text:
            text = replace(text,
                "(savedResidencyLease.bufferCount() != 748 || savedResidencyLease.byteCount() != 202252746752ULL)",
                """(savedResidencyLease.bufferCount() != 748 +
                (dense_w8a8_sep21::requiresCache(prefillRows) ? dense_w8a8_sep21::kImmutableBufferCount : 0) ||
             savedResidencyLease.byteCount() != 202252746752ULL +
                (dense_w8a8_sep21::requiresCache(prefillRows) ? dense_w8a8_sep21::Cache::plannedBytes() : 0))""")
            # Existing checkedOriginalTargetExpertResidency(), its 25-owner
            # census and host-headroom guard remain exactly inherited. The
            # composite adds only this audited 168-buffer fixed derivative.
        # Keep inherited optional FMA numerical identity intact before adding
        # this fixed cache/quantization policy, independent of dense selector.
        match = re.search(r'(      << R"\(,"target_numerical_derivative_sha256":\)" << ).*?(?=\n      << R"\()',text,re.S)
        if not match:
            raise ValueError("Dense W8A8 target derivative status anchor missing")
        expression = match.group(0)[len(match.group(1)):]
        anchor = 'json::quote('
        if expression.count(anchor) != 1:
            raise ValueError("Dense W8A8 derivative must contain exactly one quoted inherited identity")
        start = expression.index(anchor) + len(anchor)
        depth, quoted, escaped, end = 1, False, False, None
        for pos in range(start,len(expression)):
            char = expression[pos]
            if quoted:
                if escaped: escaped = False
                elif char == '\\': escaped = True
                elif char == '"': quoted = False
            elif char == '"': quoted = True
            elif char == '(': depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    end = pos
                    break
        if end is None:
            raise ValueError("Dense W8A8 inherited derivative parentheses are unbalanced")
        inherited = expression[start:end]
        status = match.group(1) + expression[:start] + 'dense_w8a8_sep21::numericalIdentity(' + inherited + ',forward_.kernelRoutes())' + expression[end:]
        status += '''
      << R"(,"dense_w8a8_prefill_enabled":)" << (dense_w8a8_sep21::requested() ? "true" : "false")
      << R"(,"dense_w8a8_numerical_alternative":true,"dense_w8a8_whole_model_qualified":false)"
      << R"(,"dense_w8a8_fixed_numerical_policy":)" << json::quote(std::string(dense_w8a8_sep21::kNumericalPolicy))
      << R"(,"dense_w8a8_cache_identity_sha256_or_scope":)" << json::quote(dense_w8a8_sep21::cacheIdentityFromRoutes(forward_.kernelRoutes()))
      << R"(,"dense_w8a8_fixed_cache_planned_bytes":)" << dense_w8a8_sep21::Cache::plannedBytes()
      << R"(,"dense_w8a8_fixed_workspace_planned_bytes":)" << dense_w8a8_sep21::Workspace::plannedBytes()
      << R"(,"dense_w8a8_projection_count":84,"dense_w8a8_immutable_buffer_count":168)"
      << R"(,"dense_w8a8_scope":"singleton main nonverification rows2048 QKV/Z/QSAQ; decoder/verify/batch/HC/router/shared/vocab/GDNout unchanged")"'''
        text = text[:match.start()] + status + text[match.end():]
    if relative == "dev/benchmarks/prefill4k_attribution.mm":
        text = replace(text, "      metal::MetalBackend backend(argv[1]);", """      (void)dense_w8a8_sep21::requested(); // Freeze strict dense selector before backend/model.
      metal::MetalBackend backend(argv[1]);""")
        text = replace(text, '      if (enabled("SPLASH_FLASH_DENSE_CACHE")) planned += FlashDenseCache::plannedBytes(weights, FlashDenseCache::defaultPrefixes(weights, true));', '''      if (dense_w8a8_sep21::requiresCache(rows)) planned += dense_w8a8_sep21::Cache::plannedBytes();
      if (enabled("SPLASH_FLASH_DENSE_CACHE")) planned += FlashDenseCache::plannedBytes(weights, FlashDenseCache::defaultPrefixes(weights, true));''')
        text = replace(text, '    if (name.starts_with("flash_moe")', '    if (name.starts_with("dense_w8a8_")) return "dense";\n    if (name.starts_with("flash_moe")')
    return text
