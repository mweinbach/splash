"""Private readonly R4 state oracle; original arithmetic/operations remain literal."""
PRIVATE='dev/benchmarks/raw_large_R4_trained_state_oracle_sep22'


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Current trained R4 source anchor drift:'+old[:100])
    return text.replace(old,new)


def transform(path,text,inspection):
    if path=='runtime/flash/FlashForward.hpp':
        anchor='  friend class FlashBatchForward;'
        if text.count(anchor)!=2:raise ValueError('Private Forward friend anchor drift')
        a,b=text.rsplit(anchor,1)
        return a+'  friend class FlashDeepPrefixOracle; // PRIVATE trained state oracle only\n'+anchor+b
    if path=='runtime/flash/FlashForward.cpp':return '#include "'+PRIVATE+'/inspect.hpp"\n'+text+'\n'+inspection
    if path!='runtime/flash/FlashWorker.mm':return text
    text='#include "'+PRIVATE+'/recorder.hpp"\n'+text
    text=once(text,'        words_((weights.descriptor().vocabularySize + 31) / 32) {}','        words_((weights.descriptor().vocabularySize + 31) / 32) {\n    if(raw_large_r4_trained_state_oracle_sep22::enabled()){\n      raw_large_r4_trained_state_oracle_sep22::check(head_&&singletonMTP_.maximumDepth==3&&!batch_&&!jointVerify_&&!jointHead_&&!batchPrefill_,"genuine singleton trained fixed3 oracle required");\n      const auto wholePlan=raw_large_r4_trained_state_oracle_sep22::Recorder::sourcePlan(capacity_);\n      trainedStateOracle_=std::make_unique<raw_large_r4_trained_state_oracle_sep22::Recorder>(backend_,governor_,forward_,*head_,wholePlan,raw_large_r4_trained_state_oracle_sep22::PolicyBinding::currentEnvironment());\n    }\n  }')
    text=once(text,'  Transport &transport_;','  std::unique_ptr<raw_large_r4_trained_state_oracle_sep22::Recorder> trainedStateOracle_;\n  Transport &transport_;')
    text=once(text,'  const auto verified = depth ? forward_.verify(*request.state, inputs)\n                              : forward_.forward(*request.state, inputs, false, true);','  const raw_large_r4_trained_state_oracle_sep22::Cookie proofCookie{id,generation,0,targetBegin,depth,static_cast<uint32_t>(inputs.size())};\n  const auto proofCarry=[&](uint32_t kept){return raw_large_r4_trained_state_oracle_sep22::WorkerCarry{targetBegin,foldedLength,request.mtpState->logicalLength(),request.emitted,kept,request.pendingToken,inputs,request.mtpFoldTokens,request.mtpFoldHidden};};\n  std::optional<raw_large_r4_trained_state_oracle_sep22::TypedTargetScope> proofScope;\n  if(trainedStateOracle_)proofScope.emplace(*trainedStateOracle_,proofCookie);\n  const auto verified = depth ? forward_.verify(*request.state, inputs)\n                              : forward_.forward(*request.state, inputs, false, true);\n  if(proofScope)proofScope->completeSuccessfulSynchronousTarget(*request.state,verified);\n  if(trainedStateOracle_)trainedStateOracle_->pending(proofCookie,*request.state,*request.mtpState,verified,proofCarry(0));')
    text=once(text,'  accepted_ += retained - 1;','  if(trainedStateOracle_)trainedStateOracle_->resolved(proofCookie,*request.state,*request.mtpState,proofCarry(retained));\n  accepted_ += retained - 1;')
    return text
