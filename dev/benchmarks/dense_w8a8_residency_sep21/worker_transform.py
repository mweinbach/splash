"""Persistent-residency hint only; no backing/math/planner/Metal changes."""
PRIVATE = "dev/benchmarks/dense_w8a8_residency_sep21"
CHANGED_FILES = {"runtime/flash/FlashForward.cpp", "runtime/flash/FlashWorker.mm", "dev/benchmarks/prefill4k_attribution.mm"}
def replace(text,old,new,count=1):
    if text.count(old) != count:
        raise ValueError(f"Dense residency frozen source drift: {old!r}")
    return text.replace(old,new)
def transform(relative,text):
    if relative not in CHANGED_FILES: return text
    text = f'#include "{PRIVATE}/worker_bridge.hpp"\n' + text
    if relative == "runtime/flash/FlashForward.cpp":
        text = replace(text,'  const bool denseW8Requested = dense_w8a8_sep21::requested();',
            '''  const bool denseW8Requested = dense_w8a8_sep21::requested();
  const bool denseW8ResidencyPrune = dense_w8a8_residency_sep21::requested();''')
        text = replace(text,'  if (impl_->denseCache) append(impl_->denseCache->persistedWeightBuffers());',
            '''  if (impl_->denseCache) append(dense_w8a8_residency_sep21::bf16PersistentOperands(
      *impl_->denseCache,bool(impl_->denseW8Cache),impl_->denseW8Requested,impl_->denseW8ResidencyPrune));''')
        text = replace(text,'  if (impl_->denseW8Cache) append(impl_->denseW8Cache->immutableWeightBuffers());',
            '''  if (dense_w8a8_residency_sep21::includeDerived(bool(impl_->denseW8Cache),
      impl_->denseW8Requested,impl_->denseW8ResidencyPrune))
    append(impl_->denseW8Cache->immutableWeightBuffers());''')
    if relative == "runtime/flash/FlashWorker.mm":
        text = replace(text,'      std::signal(SIGPIPE, SIG_IGN);',
            '''      (void)dense_w8a8_residency_sep21::requested(); // Strict hint policy frozen before paths/metadata/backend.
      std::signal(SIGPIPE, SIG_IGN);''')
        if 'savedResidencyLease.bufferCount() != 748 +' in text:
            start = text.index('        if (savedResidency.hybridExpertAdded && savedResidencyLease &&')
            end = text.index('\n          throw std::logic_error("private hybrid composite residency owner census changed");',start)
            original = text[start:end]
            text = replace(text,original,'''        if (savedResidency.hybridExpertAdded && savedResidencyLease &&
            (savedResidencyLease.bufferCount() != dense_w8a8_residency_sep21::expectedHybridOwners(
                dense_w8a8_sep21::requiresCache(prefillRows),dense_w8a8_sep21::requested(),dense_w8a8_residency_sep21::requested()) ||
             savedResidencyLease.byteCount() != dense_w8a8_residency_sep21::expectedHybridBytes(
                dense_w8a8_sep21::requiresCache(prefillRows),dense_w8a8_sep21::requested(),dense_w8a8_residency_sep21::requested())))''')
        text = replace(text,'      << R"(,"dense_w8a8_prefill_enabled":)" << (dense_w8a8_sep21::requested() ? "true" : "false")',
            '''      << R"(,"dense_w8a8_prefill_enabled":)" << (dense_w8a8_sep21::requested() ? "true" : "false")
      << R"(,"dense_w8a8_persistent_residency":{"prune_requested":)" << (dense_w8a8_residency_sep21::requested() ? "true" : "false")
      << R"(,"policy":)" << json::quote(dense_w8a8_residency_sep21::kPolicy)
      << R"(,"omitted_bf16_source_owner_count":)" << dense_w8a8_residency_sep21::plan(
          dense_w8a8_residency_sep21::hasCacheFromRoutes(forward_.kernelRoutes()),dense_w8a8_sep21::requested(),dense_w8a8_residency_sep21::requested()).omittedBF16Owners
      << R"(,"omitted_bf16_source_owner_bytes":)" << dense_w8a8_residency_sep21::plan(
          dense_w8a8_residency_sep21::hasCacheFromRoutes(forward_.kernelRoutes()),dense_w8a8_sep21::requested(),dense_w8a8_residency_sep21::requested()).omittedBF16Bytes
      << R"(,"omitted_unused_i8_owner_count":)" << dense_w8a8_residency_sep21::plan(
          dense_w8a8_residency_sep21::hasCacheFromRoutes(forward_.kernelRoutes()),dense_w8a8_sep21::requested(),dense_w8a8_residency_sep21::requested()).omittedI8Owners
      << R"(,"omitted_unused_i8_owner_bytes":)" << dense_w8a8_residency_sep21::plan(
          dense_w8a8_residency_sep21::hasCacheFromRoutes(forward_.kernelRoutes()),dense_w8a8_sep21::requested(),dense_w8a8_residency_sep21::requested()).omittedI8Bytes
      << R"(,"retained_i8_persistent_owner_count":)" << dense_w8a8_residency_sep21::plan(
          dense_w8a8_residency_sep21::hasCacheFromRoutes(forward_.kernelRoutes()),dense_w8a8_sep21::requested(),dense_w8a8_residency_sep21::requested()).retainedI8Owners
      << R"(,"retained_i8_persistent_owner_bytes":)" << dense_w8a8_residency_sep21::plan(
          dense_w8a8_residency_sep21::hasCacheFromRoutes(forward_.kernelRoutes()),dense_w8a8_sep21::requested(),dense_w8a8_residency_sep21::requested()).retainedI8Bytes
      << R"(,"ordinary_legacy_direct_bindings_transient_resident":true,"backing_freed":false,"governor_bypassed":false,"reclamation_verified":false})"''')
    if relative == "dev/benchmarks/prefill4k_attribution.mm":
        text = replace(text,'      metal::MetalBackend backend(argv[1]);',
            '''      (void)dense_w8a8_residency_sep21::requested(); // Freeze strict persistent hint policy before backend/model.
      metal::MetalBackend backend(argv[1]);''')
    return text
