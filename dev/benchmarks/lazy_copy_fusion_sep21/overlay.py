#!/usr/bin/env python3
"""Patch a copied private snapshot; never mutate runtime sources."""
from pathlib import Path


def replace(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f'expected one transform anchor in {path}: {old[:80]!r}')
    path.write_text(text.replace(old, new))


def patch(source):
    root = Path(source)
    h = root / 'runtime/flash/FlashGDNLazyRollback.hpp'
    replace(h, '[[nodiscard]] bool flashGDNLazyRollbackEnabled();',
            '[[nodiscard]] bool flashGDNLazyRollbackEnabled();\n'
            '[[nodiscard]] bool flashGDNLazyCopyFusionSep21Requested();\n'
            'inline constexpr const char *kFlashGDNLazyCopyFusionSep21Route =\n'
            '    ";private-lazy-gdn-qkv-direct-owned-tape-old-history-in-ordered-carry-exact-sep21-v1";')
    replace(h, '  [[nodiscard]] bool pending() const noexcept;',
            '  [[nodiscard]] bool pending() const noexcept;\n'
            '  [[nodiscard]] bool copyFusionEnabled() const noexcept;\n'
            '  [[nodiscard]] splash::metal::MetalBuffer rawQKVDestination(uint32_t rows, uint32_t lanes) const;')
    c = root / 'runtime/flash/FlashGDNLazyRollback.cpp'
    replace(c, '  bool inTrial = false;',
            '  bool inTrial = false;\n  const bool copyFusion = flashGDNLazyCopyFusionSep21Requested();')
    replace(c, 'FlashGDNLazyRollback::FlashGDNLazyRollback(MetalBackend &b, uint32_t r, uint32_t l)',
            '''bool flashGDNLazyCopyFusionSep21Requested() {
  const char *raw = std::getenv("SPLASH_FLASH_GDN_LAZY_COPY_FUSION_SEP21");
  if (!raw || std::string_view(raw) == "0") return false;
  if (std::string_view(raw) != "1") fail("SPLASH_FLASH_GDN_LAZY_COPY_FUSION_SEP21 must be 0 or 1");
  return true;
}
FlashGDNLazyRollback::FlashGDNLazyRollback(MetalBackend &b, uint32_t r, uint32_t l)''')
    replace(c, 'bool FlashGDNLazyRollback::pending() const noexcept { return impl_ && impl_->inTrial; }',
            '''bool FlashGDNLazyRollback::pending() const noexcept { return impl_ && impl_->inTrial; }
bool FlashGDNLazyRollback::copyFusionEnabled() const noexcept { return impl_ && impl_->copyFusion; }
MetalBuffer FlashGDNLazyRollback::rawQKVDestination(uint32_t rows, uint32_t lanes) const {
  geometry(rows, lanes);
  if (!impl_ || !impl_->copyFusion || impl_->inTrial || rows <= 1 ||
      rows > impl_->maxRows || lanes > impl_->maxLanes)
    fail("lazy GDN direct QKV requires own enabled idle multirow bounded tape");
  return impl_->view(Impl::RawQKV, uint64_t{lanes} * rows * 10240 * 2);
}''')
    replace(c, '  const std::array regions{b.qkv, b.z, b.a, b.b, b.mixed, b.decay, b.beta, b.recurrentRows,',
            '''  const bool directQKV = impl_->copyFusion && rows > 1;
  const auto directDestination = directQKV
      ? impl_->view(Impl::RawQKV, uint64_t{lanes} * rows * 10240 * 2) : MetalBuffer{};
  if (directQKV && !b.qkv.sameView(directDestination))
    fail("lazy GDN fusion requires the exact owned RawQKV destination view");
  const std::array regions{b.qkv, b.z, b.a, b.b, b.mixed, b.decay, b.beta, b.recurrentRows,''')
    replace(c, '    if (rows > 1) for (const auto &tape : impl_->buffers) disjoint(regions[i], tape);',
            '''    if (rows > 1) for (size_t plane = 0; plane < Impl::Count; ++plane) {
      // Only the precise owner RawQKV input/tape view may alias.
      if (directQKV && i == 0 && plane == Impl::RawQKV) continue;
      disjoint(regions[i], impl_->buffers[plane]);
    }''')
    replace(c, '    for (uint32_t lane = 0; lane < lanes; ++lane) {\n      const auto before = impl_->backend.view(state.convolution, uint64_t{lane} * convStride, flashGDNConvolutionLaneBytes());',
            '    if (!directQKV) for (uint32_t lane = 0; lane < lanes; ++lane) {\n      const auto before = impl_->backend.view(state.convolution, uint64_t{lane} * convStride, flashGDNConvolutionLaneBytes());')
    replace(c, '    copy(graph, b.qkv, qkv, qkv.sizeBytes());',
            '    if (!directQKV) copy(graph, b.qkv, qkv, qkv.sizeBytes());')
    replace(c, '''    graph.add("flash_gdn_convolution_carry", {qkv, state.convolution, b.diagnostics}, p,
        {40, lanes, 1}, {256, 1, 1});''',
            '''    if (directQKV) {
      const auto initialHistory = impl_->view(Impl::InitialHistory, uint64_t{lanes} * flashGDNConvolutionLaneBytes());
      graph.add("private_gdn_lazy_snapshot_convolution_carry_sep21",
          {qkv, state.convolution, initialHistory, b.diagnostics}, p,
          {40, lanes, 1}, {256, 1, 1});
    } else {
      graph.add("flash_gdn_convolution_carry", {qkv, state.convolution, b.diagnostics}, p,
          {40, lanes, 1}, {256, 1, 1});
    }''')
    replace(c, '  impl_->counters.recordTrial(rows, lanes);',
            '''  impl_->counters.recordTrial(rows, lanes);
  if (directQKV) impl_->counters.logical_raw_qkv_copy_bytes -= uint64_t{lanes} * rows * 10240 * 2;''')

    h = root / 'runtime/flash/FlashForward.hpp'
    replace(h, '  [[nodiscard]] bool lazyGDNRollbackEnabled() const noexcept;',
            '  [[nodiscard]] bool lazyGDNRollbackEnabled() const noexcept;\n'
            '  [[nodiscard]] bool lazyGDNCopyFusionSep21Enabled() const noexcept;')
    c = root / 'runtime/flash/FlashForward.cpp'
    replace(c, '  const bool lazyGDN = flashGDNLazyRollbackEnabled();',
            '  const bool lazyGDN = flashGDNLazyRollbackEnabled();\n'
            '  const bool lazyCopyFusionSep21 = flashGDNLazyCopyFusionSep21Requested();')
    replace(c, '    descriptor.validate();',
            '''    if (lazyCopyFusionSep21 && !lazyGDN)
      throw std::invalid_argument("private lazy copy fusion requires lazy GDN rollback");
    descriptor.validate();''')
    replace(c, 'bool FlashForward::lazyGDNRollbackEnabled() const noexcept { return impl_ && impl_->lazyGDN; }',
            'bool FlashForward::lazyGDNRollbackEnabled() const noexcept { return impl_ && impl_->lazyGDN; }\n'
            'bool FlashForward::lazyGDNCopyFusionSep21Enabled() const noexcept { return impl_ && impl_->lazyCopyFusionSep21; }')
    replace(c, '      (impl_->lazyGDN ? kFlashGDNLazyRollbackRoute : "") +',
            '      (impl_->lazyGDN ? kFlashGDNLazyRollbackRoute : "") +\n'
            '      (impl_->lazyCopyFusionSep21 ? kFlashGDNLazyCopyFusionSep21Route : "") +')
    replace(c, '      const auto qkv = bf(Scratch::QProjection, 10240);',
            '''      const auto qkv = verification && rows > 1 && impl_->lazyCopyFusionSep21
          ? impl_->lazyGDNRecords[layer]->rawQKVDestination(rows, 1)
          : bf(Scratch::QProjection, 10240);''')

    c = root / 'runtime/flash/FlashBatchVerify.cpp'
    replace(c, '  const bool lazyGDN = flashGDNLazyRollbackEnabled();',
            '  const bool lazyGDN = flashGDNLazyRollbackEnabled();\n'
            '  const bool lazyCopyFusionSep21 = trunk.lazyGDNCopyFusionSep21Enabled();')
    replace(c, '    validateGeometry(capacity, maximumLanes, maximumRows);',
            '''    if (lazyCopyFusionSep21 != flashGDNLazyCopyFusionSep21Requested())
      throw std::invalid_argument("private lazy copy fusion flag changed after source trunk construction");
    validateGeometry(capacity, maximumLanes, maximumRows);''')
    replace(c, '      const auto qkv = bf(Slot::Q, 10240);',
            '''      const auto qkv = impl_->lazyCopyFusionSep21 && rows > 1
          ? impl_->lazyGDNLayers[layer]->rawQKVDestination(rows, lanes)
          : bf(Slot::Q, 10240);''')

    c = root / 'runtime/flash/FlashBatchVerifyGDN.cpp'
    replace(c, 'void requireDisjoint(std::span<const Region> regions) {',
            'void requireDisjoint(std::span<const Region> regions, size_t rawAlias = SIZE_MAX) {')
    replace(c, '      if ((regions[i].writable || regions[j].writable) &&',
            '      if (!(i == 0 && j == rawAlias) &&\n          (regions[i].writable || regions[j].writable) &&')
    replace(c, '''  if (lazy)
    for (const auto &buffer : lazy->arenaBuffers())
      regions.push_back({checkedView(backend, buffer, buffer.sizeBytes(),
                                    "lazy arena"), true});''',
            '''  size_t rawAlias = SIZE_MAX;
  const bool directQKV = lazy && rows > 1 && lazy->copyFusionEnabled();
  if (directQKV && !buffers.qkv.sameView(lazy->rawQKVDestination(rows, lanes)))
    throw std::invalid_argument("Flash batch lazy fusion requires exact own raw QKV destination");
  if (lazy) {
    for (const auto &buffer : lazy->arenaBuffers()) {
      if (directQKV && buffer.sizeBytes() >= qkvBytes &&
          backend.view(buffer, 0, qkvBytes).sameView(buffers.qkv)) {
        if (rawAlias != SIZE_MAX) throw std::logic_error("duplicate lazy RawQKV allocation");
        rawAlias = regions.size();
      }
      regions.push_back({checkedView(backend, buffer, buffer.sizeBytes(), "lazy arena"), true});
    }
    if (directQKV && rawAlias == SIZE_MAX) throw std::logic_error("missing lazy RawQKV allocation");
  }''')
    replace(c, '  requireDisjoint(regions);', '  requireDisjoint(regions, rawAlias);')

    c = root / 'runtime/flash/FlashWorker.mm'
    replace(c, '      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.',
            '''      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.
      if (flashGDNLazyCopyFusionSep21Requested() &&
          (!environmentSwitch("SPLASH_FLASH_GDN_LAZY_ROLLBACK") || !environmentSwitch("SPLASH_FLASH_FUSE_GDN")))
        throw std::invalid_argument("private lazy copy fusion requires GDN_LAZY_ROLLBACK=1 and FUSE_GDN=1");''')
