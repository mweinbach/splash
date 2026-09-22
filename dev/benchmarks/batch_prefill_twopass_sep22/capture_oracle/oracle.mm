// Root-only actual batch capture. CPU/help exit before files or device.
#include "capture.hpp"
#include "CaptureBuildProvenance.hpp"
#include "flash/FlashBatchPrefill.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashGreedyGPU.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "dev/benchmarks/dense_w8a8_sep21/worker_cache.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace {
using namespace splash;
using namespace splash::flash;
using namespace splash::metal;
namespace capture=batch_qsa_capture;
namespace fs=std::filesystem;
constexpr uint32_t rows=2048,capacity=16384,hyper=10240,vocabulary=248320;
constexpr uint64_t hostPlan=3ULL<<30,margin=16ULL<<20;
constexpr const char *sourceSHA="ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e";
constexpr const char *librarySHA="1e34d3b01907acd7ad8d42532c072212b0d276336c23dc0fd49b8663711dfbaf";
constexpr const char *fixtureSHA="4985e55294b83c72cb9e51e00c40f918460b6c4f560cb5b32d4be3662e540b57";
constexpr const char *wordsSHA="0a383d21f5c784b0616d589847ca6cb04c69bf654729f2344e2b916e542f36b4";
void require(bool yes,const std::string &message){if(!yes)throw std::runtime_error(message);}
uint32_t width(const char *value){
  if(std::string_view(value)=="2")return 2;if(std::string_view(value)=="4")return 4;
  throw std::invalid_argument("actual capture width must be literal2 or4");
}
bool flag(const char *name){
  const char *value=std::getenv(name);if(!value||std::string_view(value)=="0")return false;
  if(std::string_view(value)=="1")return true;
  throw std::invalid_argument(std::string(name)+" must be0 or1");
}
void early(){
  for(const char *name:{"SPLASH_FLASH_BATCH_PREFILL","SPLASH_FLASH_BATCH_QSA_BULK_PREFILL",
      "SPLASH_FLASH_GPU_GREEDY","SPLASH_FLASH_QSA_F32","SPLASH_FLASH_QSA_MPP",
      "SPLASH_FLASH_QSA_ROW_TILES","SPLASH_FLASH_QSA_BULK_PREFILL","SPLASH_FLASH_QSA_BULK_PREFILL_SG8"})
    require(flag(name),std::string("actual old restored capture requires ")+name+"=1");
  for(const char *name:{"SPLASH_FLASH_BATCH_PREFILL_TWOPASS_SEP22","SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21",
      "SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22","SPLASH_FLASH_RAW_Q5_ROWPAIR_VERIFY_SEP22"})
    require(!flag(name),std::string("exact restored capture excludes ")+name);
  require(flashPrivateBatchBulkPrefillEnabled(),"actual legacy batch selector differs");
  (void)dense_w8a8_sep21::requiresCache(rows);
}
std::string hash(const void *raw,uint64_t bytes){
  require(raw||!bytes,"null SHA input");CC_SHA256_CTX ctx{};require(CC_SHA256_Init(&ctx),"SHA init");
  const auto *p=static_cast<const uint8_t*>(raw);
  while(bytes){const CC_LONG n=CC_LONG(std::min<uint64_t>(bytes,1ULL<<30));require(CC_SHA256_Update(&ctx,p,n),"SHA update");p+=n;bytes-=n;}
  std::array<uint8_t,CC_SHA256_DIGEST_LENGTH> d{};require(CC_SHA256_Final(d.data(),&ctx),"SHA final");
  std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto byte:d)out<<std::setw(2)<<unsigned(byte);return out.str();
}
std::string fileHash(const fs::path &path){
  const auto n=fs::file_size(path);require(n&&n<(128ULL<<20),"bounded code/fixture file extent");
  std::ifstream in(path,std::ios::binary);std::vector<uint8_t> data(n);in.read(reinterpret_cast<char*>(data.data()),std::streamsize(n));
  require(in.gcount()==std::streamsize(n),"bounded code/fixture read failed");return hash(data.data(),n);
}
std::vector<uint32_t> tokens(const fs::path &path){
  require(fileHash(path)==fixtureSHA,"original canonical2048 JSON fixture differs");
  NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  NSError *error=nil;id parsed=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(!error&&[parsed isKindOfClass:NSArray.class]&&[(NSArray*)parsed count]==rows,"exact2048 token array");
  std::vector<uint32_t> result;result.reserve(rows);
  for(id value:(NSArray*)parsed){
    require([value isKindOfClass:NSNumber.class]&&CFGetTypeID((__bridge CFTypeRef)value)!=CFBooleanGetTypeID(),"integer token, not Boolean");
    const double n=[(NSNumber*)value doubleValue];require(std::isfinite(n)&&n>=0&&n<vocabulary&&n==std::floor(n),"exact vocabulary integer token");
    result.push_back(uint32_t(n));
  }
  require(hash(result.data(),result.size()*4)==wordsSHA,"original token word identity differs");return result;
}
const char *dtype(FlashDType t){
  switch(t){case FlashDType::BF16:return "BF16";case FlashDType::F32:return "F32";case FlashDType::U32:return "U32";case FlashDType::I64:return "I64";}
  throw std::logic_error("unknown dtype");
}
uint64_t wordBytes(FlashDType t){
  switch(t){case FlashDType::BF16:return 2;case FlashDType::F32:case FlashDType::U32:return 4;case FlashDType::I64:return 8;}
  throw std::logic_error("unknown word size");
}
const char *convention(NormConvention c){
  if(c==NormConvention::OnePlusWeight)return "OnePlusWeight";if(c==NormConvention::DirectGamma)return "DirectGamma";
  throw std::logic_error("unknown actual norm convention");
}
std::string shapeJSON(const std::vector<uint64_t>&shape){
  std::ostringstream out;out<<'[';for(size_t i=0;i<shape.size();++i){if(i)out<<',';out<<shape[i];}out<<']';return out.str();
}
void finite(const void *raw,uint64_t bytes,FlashDType t,const std::string &label){
  require(raw&&bytes&&bytes%wordBytes(t)==0,"complete typed readable extent: "+label);
  if(t!=FlashDType::BF16&&t!=FlashDType::F32)return;
  const auto *p=static_cast<const uint8_t*>(raw);const auto n=wordBytes(t);
  for(uint64_t offset=0;offset<bytes;offset+=n){uint32_t bits=0;std::memcpy(&bits,p+offset,n);if(n==2)bits<<=16;
    require(std::isfinite(std::bit_cast<float>(bits)),"nonfinite complete preservation/capture plane: "+label);}
}
struct Plane{std::string name;FlashDType type;uint64_t active;std::vector<uint8_t> data;};
struct Snapshot{
  std::vector<Plane> planes;uint64_t bytes=0;
  void append(const std::string &name,FlashDType type,uint64_t active,const MetalBuffer&buffer){
    require(buffer&&buffer.storage()==BufferStorage::Shared&&buffer.contents()&&active<=buffer.sizeBytes(),"complete Shared snapshot: "+name);
    require(buffer.sizeBytes()<=hostPlan-bytes,"Host3GiB bound BEFORE snapshot vector allocation");
    finite(buffer.contents(),buffer.sizeBytes(),type,name);bytes+=buffer.sizeBytes();
    const auto *p=static_cast<const uint8_t*>(buffer.contents());planes.push_back({name,type,active,{p,p+buffer.sizeBytes()}});
  }
  uint64_t compare(size_t i,const std::string &name,FlashDType type,uint64_t active,const MetalBuffer&buffer)const{
    require(i<planes.size(),"snapshot plane missing");const auto &old=planes[i];
    require(old.name==name&&old.type==type&&old.active==active&&buffer&&buffer.storage()==BufferStorage::Shared&&buffer.contents()&&buffer.sizeBytes()==old.data.size(),"topology/active/full extent differs: "+name);
    finite(buffer.contents(),buffer.sizeBytes(),type,name);require(!std::memcmp(old.data.data(),buffer.contents(),old.data.size()),"observer changed original complete bytes: "+name);return old.data.size();
  }
};
struct Identity{std::string name;FlashDType type;MetalBuffer view;};
std::vector<Identity> fresh(FlashBatchPrefill&batch,FlashForward&trunk,const FlashRequestState&state){
  require(state.capacity()==capacity&&state.logicalLength()==0&&!state.poisoned()&&trunk.ownsState(state),"all states must be fresh healthy owned nonpending16K");
  const auto planes=batch.inspectState(state);require(planes.size()==134,"actual physical134 state plane inventory");
  std::vector<Identity> identity;uint64_t bytes=0;
  for(const auto&p:planes){require(p.buffer&&p.buffer.storage()==BufferStorage::Shared&&p.buffer.contents(),"physical state Shared view");bytes+=p.buffer.sizeBytes();identity.push_back({p.name,p.dtype,p.buffer});}
  require(bytes==FlashForward::requestStateBytes(capacity),"physical readable state extent source formula");return identity;
}
void healthy(FlashBatchPrefill&batch,FlashForward&trunk,const std::vector<FlashRequestState>&states,
    const std::vector<std::vector<Identity>>&before,const FlashBatchPrefillResult&r,uint32_t lanes){
  require(r.lanes==lanes&&r.rows==rows&&r.capacity==capacity&&r.greedyRows==lanes&&r.logicalLengths==std::vector<uint64_t>(lanes,rows)&&!r.hiddenDeliveredToDestination,"returned actual cohort geometry/publication");
  require(r.hiddenBF16.sizeBytes()==uint64_t(lanes)*rows*hyper*2&&r.logitsBF16.sizeBytes()==uint64_t(lanes)*vocabulary*2&&r.greedyResultsU32.sizeBytes()==uint64_t(lanes)*sizeof(FlashGreedyGPURowResult),"complete cohort output extents");
  for(uint32_t lane=0;lane<lanes;++lane){
    require(states[lane].capacity()==capacity&&states[lane].logicalLength()==rows&&!states[lane].poisoned()&&trunk.ownsState(states[lane]),"all states publish2048 healthy owned nonpending");
    const auto p=batch.inspectState(states[lane]);require(p.size()==before[lane].size(),"local state plane topology");
    for(size_t i=0;i<p.size();++i)require(p[i].name==before[lane][i].name&&p[i].dtype==before[lane][i].type&&p[i].buffer.sameView(before[lane][i].view),"local state view/owner replaced");
  }
  require(r.greedyResultsU32.contents(),"complete greedy records Shared");const auto*g=static_cast<const FlashGreedyGPURowResult*>(r.greedyResultsU32.contents());
  for(uint32_t lane=0;lane<lanes;++lane)(void)greedyGPUResultToken(g[lane],vocabulary);
  require(batch.canariesIntact(),"producer scratch guards changed");
}
struct Export{std::string name;FlashDType type;std::vector<uint64_t> shape;std::vector<uint8_t> data;};
void atomicText(const fs::path&path,const std::string&text){
  require(!fs::exists(path)&&!fs::exists(path.string()+".writing"),"fresh metadata path");std::ofstream out(path.string()+".writing",std::ios::binary);
  out<<text<<'\n';out.close();require(bool(out),"metadata write failed");fs::rename(path.string()+".writing",path);
}
uint64_t cpu(){
  uint64_t n=0;require(sizeof(CommandTiming)==200&&sizeof(FlashForwardCopyParams)==8,"original ABI");++n;
  require(std::endian::native==std::endian::little,"little endian fixture");++n;require(width("2")==2&&width("4")==4,"literal real widths");n+=2;
  for(const char*bad:{"0","1","3","5","02"," 2","2 ","-2",""}){bool rejected=false;try{(void)width(bad);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"invalid cohort literal");++n;}
  require(FlashForward::requestStateBytes(capacity)==582959104ULL,"exact16K physical source state bytes");++n;
  require(4*FlashForward::requestStateBytes(capacity)+uint64_t(4)*rows*hyper*2+uint64_t(4)*vocabulary*2+4*sizeof(FlashGreedyGPURowResult)<hostPlan,"B4 full snapshot+outputs Host3GiB bound");++n;
  require(capture::inputsBytes==57147392&&capture::outputBytes==25165824&&capture::inputsBytes+capture::outputBytes<capture::reservedBytes,"owned capture reservation");++n;
  require(capture::activeSession==nullptr,"defaultoff capture TLS");return n+1;
}
int gpu(char**argv){
  const fs::path directory=argv[6],report=argv[7],staging=directory.string()+".writing";
  require(!fs::exists(directory)&&!fs::exists(staging)&&!fs::exists(report)&&!fs::exists(report.string()+".failure.json"),"fresh Root directory/report");
  const auto lanes=width(argv[5]);early();require(fileHash(argv[2])==librarySHA,"exact restored current header library");
  const auto prompt=tokens(argv[4]);const auto executableSHA=fileHash(argv[0]);
  std::vector<Export> exports;std::array<NormConvention,4> conventions{};std::array<std::string,6> pipelines{};std::array<uint64_t,6> ordinals{},bufferCounts{},parameterBytes{};
  uint64_t compared=0,snapshotBytes=0,peak=0,planned=0,targetDelta=0,batchDelta=0,captureDelta=0,stateDelta=0,inputOrdinal=0,outputOrdinal=0,disjointViews=0;
  std::string source,layout,trunkRoutes,batchRoutes;std::array<uint64_t,2> submissions{};engine::MemoryGovernorSnapshot finalGov{};
  {
    MetalBackend backend(argv[2]);const auto weights=FlashWeights::load(backend,argv[3]);source=weights.sourceIdentity();layout=weights.manifestFingerprint();
    require(source==sourceSHA&&layout=="edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0"&&weights.descriptor().layerKinds[3]!=FlashLayerKind::GatedDeltaNet&&weights.descriptor().normEpsilon==1e-6&&weights.descriptor().rotaryTheta==1e7,"actual source/layer3 QSA descriptors");
    const auto mapped=backend.memoryStats().allocatedBytes;
    const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=std::max<uint64_t>(16ULL<<30,physical/10);
    require(physical>reserve,"physical host reserve");engine::MemoryGovernor governor(backend,physical-reserve,reserve);
    uint64_t targetPlan=FlashForward::workspacePlannedBytes(capacity,rows,4)+FlashForward::expertCachePlannedBytes(weights)+FlashForward::floatDenseCachePlannedBytes(weights)+FlashForward::int8HeadPlannedBytes(weights);
    if(flag("SPLASH_FLASH_DENSE_CACHE"))targetPlan+=FlashDenseCache::plannedBytes(weights,FlashDenseCache::defaultPrefixes(weights,true));
    if(flag("SPLASH_FLASH_BLOCKED_MOE"))targetPlan+=flashMoEBlockedWorkspacePlannedBytes(rows,10);
    if(dense_w8a8_sep21::requiresCache(rows))targetPlan+=dense_w8a8_sep21::Cache::plannedBytes();
    const auto batchPlan=FlashBatchPrefill::workspacePlannedBytes(capacity,4,rows),statePlan=uint64_t(lanes)*FlashForward::requestStateBytes(capacity);
    planned=targetPlan+batchPlan+statePlan+capture::reservedBytes+margin+hostPlan;
    auto hostLease=governor.tryReserve(hostPlan);require(bool(hostLease),"Host3GiB snapshot admission BEFORE large vectors");
    auto fixedLease=governor.tryReserve(targetPlan+batchPlan+capture::reservedBytes+margin);require(bool(fixedLease),"fixed target/batch/capture admission");
    {
      auto before=backend.memoryStats().allocatedBytes;FlashForward trunk(backend,weights,capacity,rows,4);targetDelta=backend.memoryStats().allocatedBytes-before;
      require(targetDelta==trunk.workspaceBytes()&&targetDelta<=targetPlan,"target actual/category ledger");
      before=backend.memoryStats().allocatedBytes;FlashBatchPrefill batch(backend,weights,trunk,capacity,4,rows);batchDelta=backend.memoryStats().allocatedBytes-before;
      require(batchDelta==batch.workspaceBytes()&&batchDelta<=batchPlan,"batch actual/category ledger");
      const std::string prefix="language_model.model.layers.3.self_attn";
      const std::array<std::string,4> names{prefix+".q_norm.weight",prefix+".k_norm.weight",prefix+".indexer.q_layernorm.weight",prefix+".indexer.k_layernorm.weight"};
      const std::array<const FlashTensor*,4> originals{&weights.tensor(names[0]),&weights.tensor(names[1]),&weights.tensor(names[2]),&weights.tensor(names[3])};
      capture::Session session(backend,originals,capture::reservedBytes);captureDelta=session.actualCharge;fixedLease->commit();
      trunkRoutes=trunk.kernelRoutes();batchRoutes=batch.kernelRoutes();Snapshot baseline;
      for(uint32_t pass=0;pass<2;++pass){
        auto stateLease=governor.tryReserve(statePlan);require(bool(stateLease),"one actual cohort state admission");before=backend.memoryStats().allocatedBytes;
        std::vector<FlashRequestState> states;states.reserve(lanes);for(uint32_t lane=0;lane<lanes;++lane)states.push_back(trunk.createState());
        auto actual=backend.memoryStats().allocatedBytes-before;stateDelta=std::max(stateDelta,actual);require(actual<=statePlan,"native owner state charge");stateLease->commit();
        std::vector<FlashRequestState*> pointers;std::vector<std::vector<Identity>> identities;std::vector<uint32_t> input;input.reserve(uint64_t(lanes)*rows);
        for(uint32_t lane=0;lane<lanes;++lane){pointers.push_back(&states[lane]);identities.push_back(fresh(batch,trunk,states[lane]));input.insert(input.end(),prompt.begin(),prompt.end());}
        require(input.size()==uint64_t(lanes)*rows,"full real lane major inputs");const auto submitted=backend.submissionCount();FlashBatchPrefillResult result;
        if(pass){capture::Install install(session);result=batch.forwardBatch(pointers,input,rows,true);}else result=batch.forwardBatch(pointers,input,rows,true);
        submissions[pass]=backend.submissionCount()-submitted;
        require(submissions[pass]==1&&backend.healthy()&&std::isfinite(result.timing.gpuSeconds)&&result.timing.gpuSeconds>0,"one healthy synchronous native submission");
        healthy(batch,trunk,states,identities,result,lanes);size_t index=0;
        const auto observe=[&](const std::string&name,FlashDType t,uint64_t active,const MetalBuffer&b){if(!pass)baseline.append(name,t,active,b);else compared+=baseline.compare(index,name,t,active,b);++index;};
        for(uint32_t lane=0;lane<lanes;++lane)for(const auto&p:batch.inspectState(states[lane]))observe("lane"+std::to_string(lane)+"."+p.name,p.dtype,p.activeBytes,p.buffer);
        observe("cohort.hidden",FlashDType::BF16,result.hiddenBF16.sizeBytes(),result.hiddenBF16);
        observe("cohort.logits",FlashDType::BF16,result.logitsBF16.sizeBytes(),result.logitsBF16);
        observe("cohort.complete_greedy",FlashDType::U32,result.greedyResultsU32.sizeBytes(),result.greedyResultsU32);
        require(index==uint64_t(lanes)*134+3,"full134state+alloutput census");
        if(!pass)snapshotBytes=baseline.bytes;else require(index==baseline.planes.size()&&compared==baseline.bytes,"full preservation comparison complete");
        require(backend.memoryStats().peakAllocatedBytes>=mapped,"native peak ledger regressed");peak=std::max(peak,backend.memoryStats().peakAllocatedBytes-mapped);require(peak<=planned-hostPlan,"actual GPU growth within admitted metal categories");
      }
      require(session.cohortValidated&&session.inputEncoded&&session.outputEncoded&&session.completed&&session.actualLanes==lanes&&session.rows==rows&&session.capacity==capacity&&session.layer==3&&session.lane==0&&session.sourceIdentity==sourceSHA&&session.disjointOtherViews>0&&session.outputInsertionDispatch==session.insertionDispatch+14,"unique coherent actual capture proof");
      capture::guardCheck(session);conventions=session.conventions;pipelines=session.legacyProducerPipelines;ordinals=session.legacyProducerOrdinals;bufferCounts=session.legacyProducerBufferCounts;parameterBytes=session.legacyProducerParameterBytes;inputOrdinal=session.insertionDispatch;outputOrdinal=session.outputInsertionDispatch;disjointViews=session.disjointOtherViews;
      for(size_t i=0;i<6;++i)require(ordinals[i]==inputOrdinal+8+i&&!pipelines[i].empty(),"six actual legacy QSA producer descriptors");
      baseline.planes.clear();baseline.planes.shrink_to_fit();uint64_t retained=0;
      const auto retain=[&](std::string name,FlashDType t,std::vector<uint64_t> shape,const MetalBuffer&b,uint64_t bytes){
        require(b&&b.storage()==BufferStorage::Shared&&b.contents()&&bytes==b.sizeBytes()&&bytes<=capture::reservedBytes-retained,"complete owned export within reservation");
        finite(b.contents(),bytes,t,name);retained+=bytes;const auto*p=static_cast<const uint8_t*>(b.contents());exports.push_back({std::move(name),t,std::move(shape),{p,p+bytes}});
      };
      const std::array<const char*,4> projectionNames{"q","k","v","index"};const std::array<uint64_t,4> widths{12288,512,512,640};
      for(size_t i=0;i<4;++i)retain(projectionNames[i],FlashDType::BF16,{rows,widths[i]},session.projected[i],session.projected[i].sizeBytes());
      retain("output",FlashDType::BF16,{rows,6144},session.output,capture::outputBytes);
      const std::array<const char*,4> normNames{"q_norm","k_norm","index_q_norm","index_k_norm"};
      for(size_t i=0;i<4;++i){
        require(session.norms[i].dtype==originals[i]->dtype&&session.norms[i].shape==originals[i]->shape&&session.norms[i].logicalBytes==originals[i]->logicalBytes&&conventions[i]==weights.normConvention(names[i]),"actual norm identity/convention");
        retain(normNames[i],session.norms[i].dtype,session.norms[i].shape,session.norms[i].buffer,session.norms[i].logicalBytes);
      }
      capture::guardCheck(session);finalGov=governor.snapshot();require(finalGov.deniedReservations==0&&finalGov.growthAllowed&&finalGov.hostMeasurementValid,"native governor pressure/admission healthy");
    }
    require(backend.memoryStats().allocatedBytes==mapped,"target/batch/state/capture owner teardown to mapped model");
    hostLease->commit();finalGov=governor.snapshot();require(finalGov.reservedBytes==0&&finalGov.deniedReservations==0&&finalGov.hostMeasurementValid&&finalGov.growthAllowed,"actual zero governor reservations and healthy host after owned teardown");
  }
  require(exports.size()==9&&compared==snapshotBytes&&snapshotBytes<hostPlan,"complete owned export/preservation schema");
  fs::create_directories(staging);std::ostringstream records;records<<'{';uint64_t payloadBytes=0;
  for(size_t i=0;i<exports.size();++i){
    const auto&p=exports[i];const std::string filename=p.name+(p.type==FlashDType::BF16?".bf16":".f32");const auto path=staging/filename;
    std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(p.data.data()),std::streamsize(p.data.size()));file.close();
    require(bool(file)&&fs::file_size(path)==p.data.size(),"owned payload write failed");payloadBytes+=p.data.size();
    if(i)records<<',';records<<json::quote(p.name)<<":{\"path\":"<<json::quote(filename)<<",\"bytes\":"<<p.data.size()<<",\"dtype\":"<<json::quote(dtype(p.type))<<",\"shape\":"<<shapeJSON(p.shape)<<",\"sha256\":"<<json::quote(hash(p.data.data(),p.data.size()))<<'}';
  }
  records<<'}';std::ostringstream out;
  out<<"{\"schema\":\"splash-current-batch-qsa-projection-capture-v1\",\"pass\":true,\"cohortValidated\":true,\"completed\":true,\"execution_complete\":true,\"sourceIdentity\":"<<json::quote(source)<<",\"layout\":"<<json::quote(layout)<<",\"actualLanes\":"<<lanes<<",\"rows\":2048,\"capacity\":16384,\"begin\":0,\"lane\":0,\"layer\":3,\"epsilon\":1e-6,\"theta\":1e7,\"files\":"<<records.str()
    <<",\"conventions\":{\"q_norm\":"<<json::quote(convention(conventions[0]))<<",\"k_norm\":"<<json::quote(convention(conventions[1]))<<",\"index_q_norm\":"<<json::quote(convention(conventions[2]))<<",\"index_k_norm\":"<<json::quote(convention(conventions[3]))<<'}'
    <<",\"graph\":{\"input_copy_ordinal\":"<<inputOrdinal<<",\"output_copy_ordinal\":"<<outputOrdinal<<",\"debug_copy_dispatches\":9,\"legacy_QSA_dispatches\":6,\"native_submissions_per_cohort\":["<<submissions[0]<<','<<submissions[1]<<"],\"legacy_producers\":[";
  for(size_t i=0;i<6;++i){if(i)out<<',';out<<"{\"ordinal\":"<<ordinals[i]<<",\"pipeline\":"<<json::quote(pipelines[i])<<",\"buffer_count\":"<<bufferCounts[i]<<",\"parameter_bytes\":"<<parameterBytes[i]<<'}';}
  out<<"]},\"preservation\":{\"same_Forward_Batch_model_process\":true,\"sequential_fresh_cohorts\":2,\"only_one_cohort_live\":true,\"all_physical_state_planes_per_lane\":134,\"full_cohort_hidden_logits_complete_greedy\":true,\"all_bytes_equal\":true,\"bytes_compared\":"<<compared<<",\"snapshot_host_bytes\":"<<snapshotBytes<<",\"other_lane_states_compared\":true,\"local_state_views_owners_unchanged\":true,\"cross_cohort_owner_pointer_equality_claimed\":false,\"whole_B1_golden_used\":false,\"Verify_tapes_Head_Worker_proved\":false,\"full_model_future_greedy_proved\":false}"
    <<",\"allocation\":{\"total_admitted_plan_bytes\":"<<planned<<",\"separate_HostRAM_reservation_bytes\":"<<hostPlan<<",\"capture_reservation_bytes\":"<<capture::reservedBytes<<",\"capture_actual_owner_charge\":"<<captureDelta<<",\"target_actual_delta\":"<<targetDelta<<",\"batch_actual_delta\":"<<batchDelta<<",\"maximum_one_cohort_state_delta\":"<<stateDelta<<",\"measured_peak_GPU_growth\":"<<peak<<",\"capture_disjoint_existing_views\":"<<disjointViews<<",\"denied_reservations\":"<<finalGov.deniedReservations<<",\"final_reserved_bytes\":"<<finalGov.reservedBytes<<",\"host_available_bytes\":"<<finalGov.hostAvailableBytes<<",\"host_reserve_bytes\":"<<finalGov.hostReserveBytes<<",\"host_headroom_bytes\":"<<finalGov.hostHeadroomBytes<<",\"host_measurement_valid\":true,\"growth_allowed\":true,\"target_state_capture_teardown_to_model_ledger\":true,\"backend_destroyed\":true}"
    <<",\"original_fixture_sha256\":"<<json::quote(fixtureSHA)<<",\"token_words_sha256\":"<<json::quote(wordsSHA)<<",\"payload_disk_bytes\":"<<payloadBytes<<",\"state_snapshot_payload_disk_bytes\":0,\"capture_clone_executable_sha256\":"<<json::quote(executableSHA)<<",\"metallib_sha256\":"<<json::quote(librarySHA)<<",\"trunk_kernel_routes\":"<<json::quote(trunkRoutes)<<",\"batch_kernel_routes\":"<<json::quote(batchRoutes)
    <<",\"scope\":\"actual old restored fresh2K B2/B4 lane0layer3 projections and legacy output; capture preservation only\",\"Root_GPU_executed\":true,\"normal_performance_measured\":false,\"new_batch_twoPass_qualified\":false,\"provenance\":"<<kCaptureBuildProvenance<<'}';
  atomicText(staging/"manifest.json",out.str());fs::rename(staging,directory);atomicText(report,out.str());
  std::cout<<"{\"pass\":true,\"manifest\":"<<json::quote((directory/"manifest.json").string())<<",\"bytes_compared\":"<<compared<<",\"backend_destroyed\":true}\n";return 0;
}
}
int main(int argc,char**argv){
  @autoreleasepool{
    try{
      if(argc==2&&(std::string_view(argv[1])=="--cpu-self-test"||std::string_view(argv[1])=="--cpu-only")){
        std::cout<<"{\"pass\":true,\"checks\":"<<cpu()<<",\"device_created\":false,\"metadata_or_payload_reads\":0,\"GPU_work\":false}\n";return 0;}
      if(argc==2&&std::string_view(argv[1])=="--help"){
        std::cout<<"oracle --gpu RESTORED_LIBRARY PACKAGE ORIGINAL2048_JSON 2|4 FRESH_CAPTURE_DIRECTORY FRESH_REPORT\nOne model/Forward/Batch, two sequential oldbulk1 fresh cohorts; Root only after full preservation; --cpu-self-test creates no device or files.\n";return 0;}
      require(argc==8&&std::string_view(argv[1])=="--gpu","explicit Root actual capture arguments");return gpu(argv);
    }catch(const std::exception&e){
      std::cerr<<"actual batch QSA capture failed: "<<e.what()<<'\n';
      if(argc==8&&std::string_view(argv[1])=="--gpu")try{
        const fs::path path=std::string(argv[7])+".failure.json";
        if(!fs::exists(path)&&!fs::exists(argv[7]))atomicText(path,std::string("{\"schema\":\"actual-batch-QSA-capture-failure-v1\",\"pass\":false,\"capture_directory_published\":")+(fs::exists(argv[6])?"true":"false")+",\"partial_staging_may_exist\":true,\"earlier_GPU_milestones_not_asserted\":true,\"reason\":"+json::quote(e.what())+'}');
      }catch(const std::exception&r){std::cerr<<"failure report write failed: "<<r.what()<<'\n';}
      return 1;
    }
  }
}
