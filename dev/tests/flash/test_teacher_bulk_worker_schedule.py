#!/usr/bin/env python3
"""Execute frozen Worker teacher scheduling blocks in a CPU-only C++ fixture.

GPU/head/transport operations are stubbed. The cookie guard and scheduling body
are extracted verbatim from the private compiled Worker; this tests callback
deletion/replacement and tail suppression without modeling matrix execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKER = ROOT / 'build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5/source/runtime/flash/FlashWorker.mm'


def braced(text, start):
    opening = text.index('{', start)
    depth = 1
    for index in range(opening + 1, len(text)):
        if text[index] == '{':
            depth += 1
        elif text[index] == '}':
            depth -= 1
            if depth == 0:
                return text[opening + 1:index], index + 1
    raise ValueError('Unclosed source block')


HARNESS = r'''
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>
using Clock=std::chrono::steady_clock;
constexpr uint32_t kHeadRows=128,kHyper=10240;
namespace metal {struct CommandTiming {double gpuSeconds=.001,wallSeconds=.002;};}
struct Buffer {uint64_t offset=0,size=2048ULL*kHyper*2;};
struct HeadState {uint64_t length=0;uint64_t logicalLength()const{return length;}};
struct Frame {uint64_t requestId=17;std::vector<uint32_t> promptTokens=std::vector<uint32_t>(2048,9);};
struct Request {Frame frame;uint64_t generation=5;bool state=true,cancelled=false;uint32_t promptOffset=2048;
 std::optional<HeadState> mtpState=HeadState{};Clock::time_point deadline=Clock::now()+std::chrono::hours(1);};
struct RequestCookie {uint64_t id=0,generation=0;};
SAME_COOKIE
struct FlashMTPTeacherBulkForward {
 static constexpr uint32_t maximumRows=2048,rowQuantum=128;
 bool fail=false;std::vector<uint32_t> submitted;
 metal::CommandTiming primeTeacherCache(HeadState &state,Buffer,std::span<const uint32_t> tokens){
   submitted.push_back(uint32_t(tokens.size()));if(fail)throw std::runtime_error("injected head failure");
   state.length+=tokens.size();return {};
 }
};
struct Backend {Buffer view(Buffer source,uint64_t offset,uint64_t bytes){
 if(offset>source.size||bytes>source.size-offset)throw std::runtime_error("bad hidden range");
 return {source.offset+offset,bytes};}};
struct Phase {double gpu=0,wall=0,lastGpu=0,lastWall=0,host=0;uint64_t rows=0,calls=0;
 void add(uint32_t count,metal::CommandTiming,double){rows+=count;++calls;}};
enum class Action {Healthy,Remove,ReplaceGeneration,ReplaceSameCookie,Cancel,Deadline,NoState,NoHead,WrongOffset,WrongLength,Stop};
struct Fixture {
 std::unique_ptr<Request> original=std::make_unique<Request>(),replacement;
 Request *live=original.get();Backend backend_;FlashMTPTeacherBulkForward bulk;
 FlashMTPTeacherBulkForward *teacherBulk_=&bulk;Action action=Action::Healthy;
 bool callbackBeforeFirst=false,safePointSaysValid=true,inFlight_=false,teacherBulkQABoundaryOpen_=false;
 bool reachedEnd=false,boundaryObserved=false;uint32_t teacherBulkQAPauseMilliseconds_=0;
 uint64_t teacherCachePrimeCalls_=0,teacherBulkCommands_=0,teacherBulkPairs_=0,teacherBulkTotalCommands_=0;
 uint64_t teacherBulkTotalPairs_=0,teacherBulkPrefixWindows_=0,teacherBulkTailCommands_=0,teacherBulkTailPairs_=0;
 uint64_t teacherBulkCompletedPairsAtBoundary_=0,teacherBulkPendingTailAtBoundary_=0;
 uint64_t teacherBulkQAPauseAttempts_=0,teacherBulkQATimeoutReleases_=0;
 uint32_t safePoints=0,statuses=0;std::vector<uint32_t> tails;Phase mtpPrime_,prefill_;
 Request *find(uint64_t id){return live&&live->frame.requestId==id?live:nullptr;}
 void publishStatus(){++statuses;if(teacherBulkQABoundaryOpen_){
   if(teacherBulkCompletedPairsAtBoundary_!=1920||teacherBulkPendingTailAtBoundary_!=127||original->mtpState->length!=1920){
      throw std::runtime_error("boundary before healthy publication");}
   boundaryObserved=true;}}
 void traceRequestCommand(const char*,const char*,RequestCookie,uint32_t,metal::CommandTiming){}
 metal::CommandTiming teacherPrime(HeadState &state,Buffer,std::span<const uint32_t> tokens){
   tails.push_back(uint32_t(tokens.size()));state.length+=tokens.size();++teacherCachePrimeCalls_;return {};}
 void callback(){
   switch(action){
   case Action::Healthy:break;
   case Action::Remove:live=nullptr;original.reset();break;
   case Action::ReplaceGeneration:case Action::ReplaceSameCookie:
     replacement=std::make_unique<Request>();replacement->generation=action==Action::ReplaceGeneration?6:5;
     live=replacement.get();original.reset();break;
   case Action::Cancel:live->cancelled=true;break;
   case Action::Deadline:live->deadline=Clock::now()-std::chrono::milliseconds(1);break;
   case Action::NoState:live->state=false;break;
   case Action::NoHead:live->mtpState.reset();break;
   case Action::WrongOffset:++live->promptOffset;break;
   case Action::WrongLength:++live->mtpState->length;break;
   case Action::Stop:break;
   }
 }
 bool safePoint(uint64_t,uint64_t){
   ++safePoints;inFlight_=false;if(safePoints==1)callback();
   return safePointSaysValid&&action!=Action::Stop;
 }
 void run(){
   Request &request=*original;const uint64_t id=request.frame.requestId,generation=request.generation;
   const Request *const teacherBulkOriginalRequest=&request;
   const uint32_t primeRows=2047,promptBegin=0;
   struct {Buffer hiddenBF16;} result;
   if(callbackBeforeFirst)callback();
INITIAL_GUARD
BULK_BODY
   reachedEnd=true;
 }
};
int main(){
 uint32_t checks=0,cases=0;auto require=[&](bool good,const char*message){++checks;if(!good)throw std::runtime_error(message);};
 try {
   {Fixture f;f.run();++cases;require(f.reachedEnd,"healthy schedule incomplete");require(f.bulk.submitted==std::vector<uint32_t>{1920},"bulk quantum wrong");
     require(f.tails==std::vector<uint32_t>{127},"healthy tail wrong");require(f.original->mtpState->length==2047,"pair state length wrong");
     require(f.teacherCachePrimeCalls_==2&&f.teacherBulkTotalCommands_==2,"actual API call counts wrong");
     require(f.teacherBulkPrefixWindows_==16&&f.teacherBulkTotalPairs_==2047,"logical coverage wrong");
     require(f.teacherBulkPairs_==1920&&f.teacherBulkTailPairs_==127,"bulk/tail metrics wrong");
     require(f.mtpPrime_.rows==2047&&f.mtpPrime_.calls==2,"phase counts wrong");}
   for(Action action:{Action::Remove,Action::ReplaceGeneration,Action::ReplaceSameCookie,Action::Cancel,Action::Deadline,
       Action::NoState,Action::NoHead,Action::WrongOffset,Action::Stop}){
     Fixture f;f.action=action;f.run();++cases;require(f.bulk.submitted==std::vector<uint32_t>{1920},"callback prevented completed bulk");
     require(f.tails.empty()&&!f.reachedEnd,"stale/cancel/deadline tail submitted");require(f.teacherBulkTotalPairs_==1920&&f.teacherCachePrimeCalls_==1,"cancelled metrics count rejected tail");}
   for(Action action:{Action::Remove,Action::ReplaceGeneration,Action::ReplaceSameCookie}){
     Fixture f;f.action=action;f.callbackBeforeFirst=true;f.run();++cases;
     require(f.bulk.submitted.empty()&&f.tails.empty()&&!f.reachedEnd,"pre-prime callback reused stale reference");}
   {Fixture f;f.action=Action::WrongLength;bool threw=false;try{f.run();}catch(const std::logic_error&){threw=true;}++cases;
     require(threw&&f.tails.empty(),"unexpected head length allowed tail");}
   {Fixture f;f.bulk.fail=true;bool threw=false;try{f.run();}catch(const std::runtime_error&){threw=true;}++cases;
     require(threw&&f.tails.empty()&&f.teacherCachePrimeCalls_==0&&f.teacherBulkTotalPairs_==0,"failed bulk published timing/metrics");}
   {Fixture f;f.teacherBulkQAPauseMilliseconds_=1;f.action=Action::Cancel;f.run();++cases;
     require(f.boundaryObserved&&!f.teacherBulkQABoundaryOpen_&&f.tails.empty(),"observable bounded QA boundary/cancel failed");
     require(f.teacherBulkQAPauseAttempts_==1&&f.teacherBulkQATimeoutReleases_==1,"QA pause/release accounting wrong");}
   {Fixture f;f.action=Action::Healthy;f.safePointSaysValid=false;f.run();++cases;require(f.tails.empty()&&!f.reachedEnd,"safePoint false ignored");}
   std::cout<<"{\"pass\":true,\"cases\":"<<cases<<",\"checks\":"<<checks
      <<",\"GPU_commands\":0,\"actual_frozen_scheduler_executed\":true,\"matrix_and_transport_are_stubs\":true}"<<std::endl;
 }catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, default=DEFAULT_WORKER)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    worker = args.worker.resolve()
    text = worker.read_text()
    cookie_start = text.index('bool sameCookie(const Request *request, RequestCookie cookie)')
    _, cookie_end = braced(text, cookie_start)
    cookie = text[cookie_start:cookie_end]
    tick = text.index('  void tick(Request &request) {')
    prime = text.index('        if(teacherBulk_) {\n          const RequestCookie cookie{id,generation};', tick)
    body, _ = braced(text, prime)
    guard_start = text.index('      if(teacherBulk_) {\n        const auto *live=find(id);', tick)
    _, guard_end = braced(text, guard_start)
    guard = text[guard_start:guard_end]
    source = HARNESS.replace('SAME_COOKIE', cookie).replace('INITIAL_GUARD', guard).replace('BULK_BODY', body)
    with tempfile.TemporaryDirectory(prefix='teacher-worker-schedule-') as directory:
        folder = Path(directory)
        cpp = folder / 'fixture.cpp'
        binary = folder / 'fixture'
        cpp.write_text(source)
        build = subprocess.run(['xcrun', 'clang++', '-std=c++20', '-O0', '-Wall', '-Wextra', '-Werror',
                                str(cpp), '-o', str(binary)], text=True, capture_output=True)
        if build.returncode:
            raise RuntimeError(build.stderr)
        run = subprocess.run([str(binary)], text=True, capture_output=True, timeout=10)
        if run.returncode:
            raise RuntimeError(run.stderr)
        result = json.loads(run.stdout)
    result.update(worker_path=str(worker), worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
                  extracted_cookie_sha256=hashlib.sha256(cookie.encode()).hexdigest(),
                  extracted_initial_guard_sha256=hashlib.sha256(guard.encode()).hexdigest(),
                  extracted_bulk_scheduler_sha256=hashlib.sha256(body.encode()).hexdigest(),
                  fixture_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  runtime_Worker_recovery_tested=False)
    if args.output:
        output = args.output.resolve()
        if output.exists():
            raise ValueError('Choose fresh output')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
