"""Source-only overlay. Original arithmetic/state bodies remain literal."""
PRIVATE = 'dev/benchmarks/R5_raw_current_input_capture_sep22'


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Current R5 capture source anchor drift: ' + old[:100])
    return text.replace(old, new)


def transform(path, text, inspection):
    if path == 'runtime/flash/FlashForward.hpp':
        anchor = '  friend class FlashBatchForward;'
        if text.count(anchor) != 2:
            raise ValueError('Private Forward friend anchor drift')
        before, after = text.rsplit(anchor, 1)
        return before + '  friend class FlashDeepPrefixOracle; // PRIVATE capture clone only\n' + anchor + after
    if path not in ('runtime/flash/FlashForward.cpp', 'runtime/flash/FlashWorker.mm'):
        return text
    text = '#include "' + PRIVATE + '/proof.hpp"\n' + text
    if path.endswith('FlashForward.cpp'):
        text = once(text, '        else {\n          if (rawQ4Rowpair', '        else {\n          r5_raw_current_capture_sep22::captureInput(graph,prefix,input,projection,output,diagnostics,rows,verification,singletonMain,floatDenseCache->contains(prefix),!tile);\n          if (rawQ4Rowpair')
        text = once(text, '    timing = impl_->backend.submitCommand(graph.dispatches());\n    uint32_t status = 0;\n', '    r5_raw_current_capture_sep22::recordGraph(graph,verification,rows,begin);\n    timing = impl_->backend.submitCommand(graph.dispatches());\n    uint32_t status = 0;\n')
        return text + '\n' + inspection
    text = once(text, '      raw_q4_verify_sep22::validateDependencies(); // Freeze strict rowpair before paths/model/backend.', '      r5_raw_current_capture_sep22::validateStartup(); // Private source-only diagnostic admission.\n      raw_q4_verify_sep22::validateDependencies(); // Freeze strict rowpair before paths/model/backend.')
    text = once(text, '        words_((weights.descriptor().vocabularySize + 31) / 32) {}', '        words_((weights.descriptor().vocabularySize + 31) / 32) {\n    if(r5_raw_current_capture_sep22::Config::frozenConfig().proof){\n      r5_raw_current_capture_sep22::require(head_&&singletonMTP_.explicitOverride&&singletonMTP_.maximumDepth==4&&!batch_&&!jointVerify_&&!jointHead_&&!batchPrefill_,"R5 capture requires genuine singleton trained fixed4 source");\n      const auto wholeCapturePlan=r5_raw_current_capture_sep22::Proof::sourcePlan(capacity_); // BEFORE capture owner/host/slot allocations.\n      rawCapture_=std::make_unique<r5_raw_current_capture_sep22::Owner>(backend_,governor_,FlashDeepPrefixOracle::captureInventory(forward_));\n      for(const auto &slot:rawCapture_->slots)if(slot.base)FlashDeepPrefixOracle::validateCaptureDestination(forward_,slot.base);\n      rawProof_=std::make_unique<r5_raw_current_capture_sep22::Proof>(*rawCapture_,forward_,wholeCapturePlan);\n    }\n  }')
    text = once(text, '  Transport &transport_;', '  std::unique_ptr<r5_raw_current_capture_sep22::Owner> rawCapture_;\n  std::unique_ptr<r5_raw_current_capture_sep22::Proof> rawProof_;\n  Transport &transport_;')
    text = once(text, '  const auto verified = depth ? forward_.verify(*request.state, inputs)\n                              : forward_.forward(*request.state, inputs, false, true);', '  std::optional<r5_raw_current_capture_sep22::Scope> rawScope;\n  if(rawCapture_)rawScope.emplace(*rawCapture_,id,generation,depth,static_cast<uint32_t>(inputs.size()));\n  const auto verified = depth ? forward_.verify(*request.state, inputs)\n                              : forward_.forward(*request.state, inputs, false, true);\n  if(rawScope)rawScope->complete();\n  if(rawProof_)rawProof_->verified(*request.state,verified,id,generation,depth,static_cast<uint32_t>(inputs.size()));')
    text = once(text, '  const auto committed = depth ? forward_.commitVerify(*request.state, retained) : metal::CommandTiming{};', '  const auto committed = depth ? forward_.commitVerify(*request.state, retained) : metal::CommandTiming{};\n  if(rawProof_)rawProof_->committed(*request.state,retained,id,generation);')
    return text
