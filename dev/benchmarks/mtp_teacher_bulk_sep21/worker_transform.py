"""Whole Worker integration; only singleton teacher scheduling changes."""
from pathlib import Path
import importlib.util
PRIVATE='dev/benchmarks/mtp_teacher_bulk_sep21'
def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Teacher Worker anchor differs: '+old[:100])
    return text.replace(old,new,1)
def transform(relative,text):
    if relative in ('runtime/flash/FlashMTP.cpp','runtime/flash/FlashMTP.hpp'):
        modulePath=Path(__file__).with_name('prepare.py');spec=importlib.util.spec_from_file_location('teacher_methods',modulePath);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.transform(relative,text)
    if relative!='runtime/flash/FlashWorker.mm':return text
    text=f'#include "{PRIVATE}/bulk.hpp"\n#include "{PRIVATE}/policy.hpp"\n'+text
    text=once(text,'bool idleMaintenanceRequested = false, std::string idleMaintenanceFailureReason = {})',
        'bool idleMaintenanceRequested = false, std::string idleMaintenanceFailureReason = {},\n         FlashMTPTeacherBulkForward *teacherBulk = nullptr, uint32_t teacherBulkQAPauseMilliseconds = 0)')
    text=once(text,'        prefillRows_(prefillRows),','        prefillRows_(prefillRows), teacherBulk_(teacherBulk), teacherBulkQAPauseMilliseconds_(teacherBulkQAPauseMilliseconds),')
    text=once(text,'  uint64_t teacherCachePrimeCalls_ = 0;',
        '''  uint64_t teacherCachePrimeCalls_ = 0;
  bool teacherBulkQABoundaryOpen_ = false;
  uint64_t teacherBulkCommands_ = 0, teacherBulkPairs_ = 0, teacherBulkPrefixWindows_ = 0;
  uint64_t teacherBulkTotalCommands_ = 0, teacherBulkTotalPairs_ = 0, teacherBulkTailCommands_ = 0, teacherBulkTailPairs_ = 0;
  uint64_t teacherBulkCompletedPairsAtBoundary_ = 0, teacherBulkPendingTailAtBoundary_ = 0;''')
    text=once(text,'  uint64_t teacherBulkCompletedPairsAtBoundary_ = 0, teacherBulkPendingTailAtBoundary_ = 0;',
        '  uint64_t teacherBulkCompletedPairsAtBoundary_ = 0, teacherBulkPendingTailAtBoundary_ = 0;\n  uint64_t teacherBulkQAPauseAttempts_ = 0, teacherBulkQATimeoutReleases_ = 0;')
    text=once(text,'  uint32_t prefillRows_ = kDefaultPrefillRows;',
        '  uint32_t prefillRows_ = kDefaultPrefillRows;\n  FlashMTPTeacherBulkForward *teacherBulk_ = nullptr;\n  uint32_t teacherBulkQAPauseMilliseconds_ = 0;')
    begin=text.index('        for (size_t primeBegin = 0; primeBegin < primeRows; primeBegin += kHeadRows) {')
    end=text.index('        request.mtpPriming = false;',begin)
    original=text[begin:end]
    bulk='''        if(teacherBulk_) {
          const RequestCookie cookie{id,generation};
          const Request *const originalRequest=&request;
          const auto expectedPromptOffset=request.promptOffset;
          size_t primeBegin=0;
          while(primeBegin<primeRows) {
            auto *member=find(cookie.id);
            if(!sameCookie(member,cookie) || member!=originalRequest || !member->state || !member->mtpState ||
                member->promptOffset!=expectedPromptOffset || member->cancelled || Clock::now()>=member->deadline)return;
            const size_t remaining=primeRows-primeBegin;
            const size_t complete=std::min<size_t>(FlashMTPTeacherBulkForward::maximumRows,
                remaining/FlashMTPTeacherBulkForward::rowQuantum*FlashMTPTeacherBulkForward::rowQuantum);
            const size_t primeCount=complete?complete:std::min<size_t>(kHeadRows,remaining);
            const uint64_t expectedHeadLength=member->mtpState->logicalLength()+primeCount;
            const auto hidden=backend_.view(result.hiddenBF16,uint64_t{primeBegin}*kHyper*2,uint64_t{primeCount}*kHyper*2);
            inFlight_=true;publishStatus();const auto primeBegan=Clock::now();
            const auto next=std::span(member->frame.promptTokens).subspan(promptBegin+primeBegin+1,primeCount);
            metal::CommandTiming timing;
            if(complete) {
              timing=teacherBulk_->primeTeacherCache(*member->mtpState,hidden,next);
              ++teacherCachePrimeCalls_;
              ++teacherBulkCommands_;teacherBulkPairs_+=primeCount;
            } else timing=teacherPrime(*member->mtpState,hidden,next);
            ++teacherBulkTotalCommands_;teacherBulkTotalPairs_+=primeCount;
            teacherBulkPrefixWindows_+=(primeCount+kHeadRows-1)/kHeadRows;
            if(!complete){++teacherBulkTailCommands_;teacherBulkTailPairs_+=primeCount;}
            traceRequestCommand("prompt_head_priming","mtp_head",cookie,static_cast<uint32_t>(primeCount),timing);
            const auto host=std::chrono::duration<double>(Clock::now()-primeBegan).count();
            mtpPrime_.add(static_cast<uint32_t>(primeCount),timing,host);
            prefill_.gpu+=timing.gpuSeconds;prefill_.wall+=timing.wallSeconds;
            prefill_.lastGpu+=timing.gpuSeconds;prefill_.lastWall+=timing.wallSeconds;prefill_.host+=host;
            primeBegin+=primeCount;
            // Private QA exposes a deterministic successful-bulk boundary.
            // Reader controls queue during the bounded pause, then the actual
            // safePoint drains cancellation/deadline before any tail/ref reuse.
            if(complete && teacherBulkQAPauseMilliseconds_) {
              ++teacherBulkQAPauseAttempts_;inFlight_=false;
              teacherBulkQABoundaryOpen_=true;
              teacherBulkCompletedPairsAtBoundary_=primeBegin;
              teacherBulkPendingTailAtBoundary_=primeRows-primeBegin;publishStatus();
              std::this_thread::sleep_for(std::chrono::milliseconds(teacherBulkQAPauseMilliseconds_));
              ++teacherBulkQATimeoutReleases_;
              teacherBulkQABoundaryOpen_=false;
            }
            if(!safePoint(cookie.id,cookie.generation))return;
            member=find(cookie.id);
            if(!sameCookie(member,cookie) || member!=originalRequest || !member->state || !member->mtpState ||
                member->promptOffset!=expectedPromptOffset || member->cancelled || Clock::now()>=member->deadline)return;
            if(member->mtpState->logicalLength()!=expectedHeadLength)
              throw std::logic_error("teacher bulk completed cache length differs");
          }
          auto *member=find(cookie.id);
          if(!sameCookie(member,cookie) || member!=originalRequest)return;
        } else {
'''+original+'''        }
'''
    text=text[:begin]+bulk+text[end:]
    text=once(text,'      << R"(,"teacher_cache_only_priming_calls":)" << teacherCachePrimeCalls_',
        '''      << R"(,"teacher_cache_only_priming_calls":)" << teacherCachePrimeCalls_
      << R"(,"singleton_teacher_bulk":{"requested":)" << (teacherBulk_?"true":"false")
      << R"(,"maximum_bulk_rows":)" << (teacherBulk_?2048:0)
      << R"(,"allocated_workspace_bytes":)" << (teacherBulk_?teacherBulk_->workspaceBytes():0)
      << R"(,"planned_workspace_bytes":)" << (teacherBulk_?FlashMTPTeacherBulkForward::plannedBytes:0)
      << R"(,"completed_bulk_commands":)" << teacherBulkCommands_ << R"(,"completed_bulk_pairs":)" << teacherBulkPairs_
      << R"(,"completed_teacher_commands":)" << teacherBulkTotalCommands_ << R"(,"completed_pairs":)" << teacherBulkTotalPairs_
      << R"(,"completed_tail_commands":)" << teacherBulkTailCommands_ << R"(,"completed_tail_pairs":)" << teacherBulkTailPairs_
      << R"(,"completed_original128_cache_prefix_windows":)" << teacherBulkPrefixWindows_
      << R"(,"completed_original128_cache_prefix_rows":)" << teacherBulkTotalPairs_
      << R"(,"legacy_teacher_call_scope":"successful actual arena/API invocations; original128 logical cache prefixes counted separately","numeric_derivative_changed":false,"cache_math":"original trained head global10240 RMS then four fc_hidden streams; unchanged M32/M16 descriptors; original chronological128 cache prefixes"})"
      << R"(,"singleton_teacher_bulk_qa":{"pause_milliseconds":)" << teacherBulkQAPauseMilliseconds_
      << R"(,"boundary_open":)" << (teacherBulkQABoundaryOpen_?"true":"false")
      << R"(,"completed_pairs_at_boundary":)" << teacherBulkCompletedPairsAtBoundary_
      << R"(,"pending_tail_pairs_at_boundary":)" << teacherBulkPendingTailAtBoundary_ << '}' ''')
    text=once(text,'      << R"(,"pending_tail_pairs_at_boundary":)" << teacherBulkPendingTailAtBoundary_ << \'}\' ',
        '''      << R"(,"pending_tail_pairs_at_boundary":)" << teacherBulkPendingTailAtBoundary_
      << R"(,"attempts":)" << teacherBulkQAPauseAttempts_
      << R"(,"timeout_releases":)" << teacherBulkQATimeoutReleases_ << R"(,"early_releases":0})" ''')
    text=once(text,'      std::signal(SIGPIPE, SIG_IGN);',
        '''      const bool singletonTeacherBulkRequested=teacher_bulk_sep21::parse(std::getenv(teacher_bulk_sep21::flag));
      const char *teacherBulkPause=std::getenv("SPLASH_FLASH_TEACHER_BULK_QA_PAUSE_MS");
      const uint32_t teacherBulkQAPauseMilliseconds=!teacherBulkPause || std::string_view(teacherBulkPause)=="0"?0:
          std::string_view(teacherBulkPause)=="500"?500:throw std::invalid_argument("private teacher QA pause must be 0 or 500 ms");
      if(teacherBulkQAPauseMilliseconds && !singletonTeacherBulkRequested)
        throw std::invalid_argument("teacher QA pause requires singleton bulk policy");
      teacher_bulk_sep21::validate(singletonTeacherBulkRequested,{environmentSwitch("SPLASH_FLASH_MTP"),environmentSwitch("SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY"),
          environmentSwitch("SPLASH_FLASH_DENSE_CACHE"),environmentSwitch("SPLASH_FLASH_MTP_QSA_F32"),environmentSwitch("SPLASH_FLASH_MTP_QSA_MPP")});
      std::signal(SIGPIPE, SIG_IGN);''')
    text=once(text,'      const bool teacherCacheOnlyEnabled = environmentSwitch("SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY");',
        '''      const bool teacherCacheOnlyEnabled = environmentSwitch("SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY");
''')
    marker='      std::optional<FlashBatchForward> batch;'
    construction='''      std::optional<FlashMTPTeacherBulkForward> teacherBulk;
      if(singletonTeacherBulkRequested) {
        metal::AllocationFailure failure=metal::AllocationFailure::None;
        auto reservation=governor.tryReserve(FlashMTPTeacherBulkForward::plannedBytes,&failure);
        if(!reservation)throw metal::MetalAllocationError("singleton teacher bulk cannot reserve workspace",failure);
        const auto before=backend.memoryStats().allocatedBytes;
        teacherBulk.emplace(*head);
        if(teacherBulk->workspaceBytes()>FlashMTPTeacherBulkForward::plannedBytes ||
            metal::allocationDelta(before,backend.memoryStats().allocatedBytes)>FlashMTPTeacherBulkForward::plannedBytes)
          throw std::runtime_error("singleton teacher bulk exceeds reserved workspace");
        reservation->commit();
      }
'''
    text=once(text,marker,construction+marker)
    text=once(text,'                    std::move(idleMaintenanceFailureReason));',
        '                    std::move(idleMaintenanceFailureReason), teacherBulk?&*teacherBulk:nullptr,teacherBulkQAPauseMilliseconds);')
    text=once(text,'      << R"(,"loaded_model_layout_sha256":)" << json::quote(weights_.manifestFingerprint())',
        '''      << R"(,"loaded_model_layout_sha256":)" << json::quote(weights_.manifestFingerprint())
      << R"(,"singleton_teacher_bulk_enabled":)" << (teacherBulk_?"true":"false")
      << R"(,"singleton_teacher_bulk_schema":)" << (teacherBulk_?json::quote("singleton-teacher-original-cache-bulk2048-prefix128-v1"):"null")
      << R"(,"singleton_teacher_bulk_source_sha256":)" << (teacherBulk_?json::quote("99b7f20ae3d78ad1b81a4fe47cef2a04c0ac327eea9732174e1d6e86c1bb7e1a"):"null")''')
    text=once(text,'  void tick(Request &request) {\n    const uint64_t id = request.frame.requestId;\n    const uint64_t generation = request.generation;',
        '  void tick(Request &request) {\n    const uint64_t id = request.frame.requestId;\n    const uint64_t generation = request.generation;\n    const Request *const teacherBulkOriginalRequest=&request;')
    text=once(text,'const Request *const originalRequest=&request;','const Request *const originalRequest=teacherBulkOriginalRequest;')
    text=once(text,'      if (!safePoint(id, generation)) return;\n      if (request.mtpState) {\n        const auto primeRows',
        '''      if (!safePoint(id, generation)) return;
      if(teacherBulk_) {
        const auto *live=find(id);
        if(!sameCookie(live,{id,generation}) || live!=teacherBulkOriginalRequest)return;
      }
      if (request.mtpState) {
        const auto primeRows''')
    return text
