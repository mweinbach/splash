// Root-only GPU oracle. CPU mode and compilation submit zero GPU commands.
#define main prefill_qsa_original_oracle_main
#include "base_oracle.mm"
#undef main
#include "bulk.hpp"
#include "twopass.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryPlan.hpp"
#include <tuple>
namespace {
namespace metal=splash::metal;
using namespace splash::flash::prefill4k;
constexpr uint64_t kStandaloneReservation=2ULL*1024*1024*1024;
constexpr double kSourceBF16L2=1e-4,kSourceRawL2=2e-5,kSourceCosine=.99999999;
constexpr double kScoreAbs=2e-5,kScoreRelative=2e-6,kRawAbs=2e-6,kRawRelative=2e-5;
FlashTensor normTensor(MetalBackend &b,uint32_t width,bool f32,uint64_t seedValue) {
  FlashTensor t;t.dtype=f32?FlashDType::F32:FlashDType::BF16;t.shape={width};t.logicalBytes=uint64_t(width)*(f32?4:2);
  t.buffer=b.allocateBuffer(t.logicalBytes,BufferStorage::Shared,"two-pass immutable norm");
  for (uint32_t i=0;i<width;++i) {const float v=.2f+seeded(i,seedValue,.1f);
    if (f32) static_cast<float *>(t.buffer.contents())[i]=v;else static_cast<uint16_t *>(t.buffer.contents())[i]=bf16(v);}
  return t;
}
struct Input final {
  MetalBuffer q,k,v,index,diag;FlashTensor qn,kn,iqn,ikn;Guarded output;uint32_t rows;
  Input(MetalBackend &b,uint32_t n,uint64_t seedValue):qn(normTensor(b,256,true,111)),kn(normTensor(b,256,false,112)),
      iqn(normTensor(b,128,false,113)),ikn(normTensor(b,128,true,114)),output(b,uint64_t(n)*6144),rows(n) {
    auto fill=[&](uint32_t width,uint64_t salt,float scale) {auto x=b.allocateBuffer(uint64_t(n)*width*2,BufferStorage::Shared,"two-pass projected input");
      auto *p=static_cast<uint16_t *>(x.contents());for (uint64_t i=0;i<uint64_t(n)*width;++i) p[i]=bf16(seeded(i,seedValue+salt,scale));return x;};
    q=fill(12288,1,2);k=fill(512,2,2);v=fill(512,3,.75);index=fill(640,4,1.5);
    diag=b.allocateBuffer(4,BufferStorage::Shared,"two-pass diagnostics");*static_cast<uint32_t *>(diag.contents())=kSticky;
  }
  FlashQSAFastInputs input() {return {q,k,v,index,&qn,&kn,&iqn,&ikn,output.view,diag,{},NormConvention::OnePlusWeight,
      NormConvention::OnePlusWeight,NormConvention::OnePlusWeight,NormConvention::OnePlusWeight,1e-6,1e7};}
  std::string hash() const {SHA256 h;for (const auto &x:{q,k,v,index,qn.buffer,kn.buffer,iqn.buffer,ikn.buffer}) h.add(x);return h.finish();}
};
struct GuardedBytes final {
  MetalBuffer base,view;uint64_t bytes;static constexpr uint64_t guard=256;
  GuardedBytes(MetalBackend &b,uint64_t n):bytes(n) {base=b.allocateBuffer(n+2*guard,BufferStorage::Shared,"two-pass external guarded plane");
    std::memset(base.contents(),0xa5,n+2*guard);view=b.view(base,guard,n);}
  void check() const {const auto *p=static_cast<const uint8_t *>(base.contents());
    for (uint64_t i=0;i<guard;++i) require(p[i]==0xa5&&p[guard+bytes+i]==0xa5,"Two-pass external plane guard changed");}
};
struct StateGuards final {
  GuardedBytes keys,values,index,pooled,positions;
  StateGuards(MetalBackend &backend,FlashQSAState &s):keys(backend,s.keys.sizeBytes()),values(backend,s.values.sizeBytes()),
      index(backend,s.rawIndexKeys.sizeBytes()),pooled(backend,s.pooledKeys.sizeBytes()),positions(backend,s.indexPositions.sizeBytes()) {
    s.keys=keys.view;s.values=values.view;s.rawIndexKeys=index.view;s.pooledKeys=pooled.view;s.indexPositions=positions.view;
  }
  void check() const {keys.check();values.check();index.check();pooled.check();positions.check();}
};
std::array<MetalBuffer,5> statePlanes(const FlashQSAState &s) {return {s.keys,s.values,s.rawIndexKeys,s.pooledKeys,s.indexPositions};}
void poisonPadding(const FlashQSAState &s) {for (const auto &x:statePlanes(s)) std::memset(x.contents(),0xa5,x.sizeBytes());
  for (const auto &x:{s.keys,s.values}) {auto *p=static_cast<uint16_t *>(x.contents());std::fill_n(p,x.sizeBytes()/2,kSentinel);}}
void equalState(const FlashQSAState &a,const FlashQSAState &b) {auto x=statePlanes(a),y=statePlanes(b);
  for (uint32_t i=0;i<5;++i) require(x[i].sizeBytes()==y[i].sizeBytes()&&!std::memcmp(x[i].contents(),y[i].contents(),x[i].sizeBytes()),"Two-pass changed five-plane persistent cache bytes");}
std::string buffersHash(std::initializer_list<MetalBuffer> buffers) {SHA256 h;for (const auto &b:buffers) h.add(b);return h.finish();}
void append(metal::CommandGraph &g,const metal::ComputeDispatch &d) {
  std::vector<MetalBuffer> buffers;for (const auto &b:d.buffers) {require(b.index==buffers.size(),"oracle binding order drift");buffers.push_back(b.buffer);}
  require(d.bytes.size()==1,"oracle dispatch byte binding drift");
  if (d.bytes[0].sizeBytes==sizeof(FlashQSAParams)) {FlashQSAParams p{};std::memcpy(&p,d.bytes[0].data,sizeof(p));g.add(d.pipelineName,buffers,p,d.threadgroups,d.threadsPerThreadgroup);}
  else {require(d.bytes[0].sizeBytes==sizeof(FlashQSAFastParams),"oracle dispatch ABI drift");FlashQSAFastParams p{};std::memcpy(&p,d.bytes[0].data,sizeof(p));g.add(d.pipelineName,buffers,p,d.threadgroups,d.threadsPerThreadgroup);}
}
CommandGraph range(const CommandGraph &g,uint32_t first,uint32_t stop) {CommandGraph result;const auto all=g.dispatches();
  require(first<=stop&&stop<=all.size(),"oracle graph range is invalid");for (uint32_t i=first;i<stop;++i) append(result,all[i]);return result;}
struct FloatError final {
  uint64_t n=0,bits=0,nonfinite=0,signFlips=0;double square=0,referenceSquare=0,actualSquare=0,dot=0,maxAbs=0;
  void add(float a,float b) {++n;bits+=std::bit_cast<uint32_t>(a)!=std::bit_cast<uint32_t>(b);
    if (!std::isfinite(a)||!std::isfinite(b)) {++nonfinite;return;}const double d=double(a)-b;square+=d*d;referenceSquare+=double(b)*b;
    actualSquare+=double(a)*a;dot+=double(a)*b;maxAbs=std::max(maxAbs,std::abs(d));signFlips+=std::abs(b)>kRawAbs&&std::signbit(a)!=std::signbit(b);}
  double l2() const {return std::sqrt(square/std::max(referenceSquare,1e-30));}
  double cosine() const {return dot/std::sqrt(std::max(actualSquare*referenceSquare,1e-60));}
  void write(std::ostream &o) const {o<<"{\"elements\":"<<n<<",\"f32_bit_mismatches\":"<<bits<<",\"nonfinite\":"<<nonfinite
    <<",\"relative_l2\":"<<l2()<<",\"cosine\":"<<cosine()<<",\"max_abs\":"<<maxAbs<<",\"sign_flips_above_absolute_floor\":"<<signFlips<<'}';}
};
struct Certificate final {
  uint64_t cells=0,failures=0,signFailures=0;double maxAbs=0,maxNormalized=0;
  void add(double a,double b,double absolute,double relative) {++cells;const double d=std::abs(a-b),bound=absolute+relative*std::abs(b);
    failures+=!std::isfinite(a)||!std::isfinite(b)||d>bound;signFailures+=std::abs(b)>absolute&&std::signbit(a)!=std::signbit(b);
    maxAbs=std::max(maxAbs,d);maxNormalized=std::max(maxNormalized,d/bound);}
  void write(std::ostream &o) const {o<<"{\"cells\":"<<cells<<",\"absolute_relative_envelope_failures\":"<<failures<<",\"sign_failures_above_absolute_floor\":"<<signFailures
    <<",\"max_abs\":"<<maxAbs<<",\"max_error_over_envelope\":"<<maxNormalized<<'}';}
};
struct ReferenceRow {uint32_t row,head;std::vector<double> probability;std::array<double,8> raw;};
constexpr std::array<uint32_t,8> columns{0,1,31,63,64,127,191,255};
std::vector<ReferenceRow> f64Reference(const TwoPassWorkspace &w,const FlashQSAState &state,Certificate &scores) {
  const auto *q=static_cast<const uint16_t *>(w.prepared.queries.contents()),*k=static_cast<const uint16_t *>(state.keys.contents()),*v=static_cast<const uint16_t *>(state.values.contents());
  const auto *actual=static_cast<const float *>(w.scoresAndProbabilities.contents());std::vector<ReferenceRow> result;
  for (uint32_t row:{0u,1u,63u,64u,127u,128u,511u,512u,1023u,1535u,2047u}) for (uint32_t head:{0u,5u,11u,12u,23u}) {
    ReferenceRow r{row,head,std::vector<double>(row+1),{}};double maximum=-INFINITY;
    for (uint32_t token=0;token<=row;++token) {double sum=0;
      for (uint32_t d=0;d<256;++d) sum+=double(number(q[(uint64_t(row)*24+head)*256+d]))*number(k[(uint64_t(token)*2+head/12)*256+d]);
      r.probability[token]=sum*.0625;maximum=std::max(maximum,r.probability[token]);
      scores.add(actual[(uint64_t(head/12)*24576+row*12+head%12)*2048+token],r.probability[token],kScoreAbs,kScoreRelative);}
    double denominator=0;for (auto &s:r.probability) {s=std::exp(s-maximum);denominator+=s;}
    for (auto &p:r.probability) p/=denominator;
    for (uint32_t i=0;i<columns.size();++i) for (uint32_t token=0;token<=row;++token)
      r.raw[i]+=r.probability[token]*number(v[(uint64_t(token)*2+head/12)*256+columns[i]]);
    result.push_back(std::move(r));
  }
  return result;
}
uint64_t cpuChecks() {
  uint64_t checks=0;require(twoPassPlannedBytes()==509607936&&twoPassExtraBytes()==478150656,"two-pass exact byte plan changed");checks+=2;
  for (const auto &[begin,rows,capacity,verify,expected]:std::vector<std::tuple<uint32_t,uint32_t,uint32_t,bool,bool>>{
      {0,2048,2048,false,true},{0,2048,262144,false,true},{1,2048,4096,false,false},{0,2047,4096,false,false},
      {0,2048,2047,false,false},{0,2048,262145,false,false},{0,2048,4096,true,false},{0,4096,8192,false,false}}) {
    require(twoPassGeometry(begin,rows,capacity,verify)==expected,"two-pass CPU selector failure");++checks;}
  for (uint32_t value=0;value<65536;++value) if (std::isfinite(number(uint16_t(value)))) {require(bf16(number(uint16_t(value)))==value,"two-pass BF16 roundtrip");++checks;}
  // Per-row2048 score cells are retained as8 private values/thread; every cell has one writer.
  std::array<uint32_t,2048> visits{};for (uint32_t tid=0;tid<256;++tid) for (uint32_t i=0;i<8;++i) ++visits[tid+i*256];
  for (auto n:visits) {require(n==1,"softmax private ownership not bijective");++checks;}
  require(sizeof(CommandTiming)==200,"timing ABI changed");++checks;return checks;
}
}
int main(int argc,char **argv) {
  @autoreleasepool {
    std::string failureMetrics;
    try {
      if (argc==2&&std::string(argv[1])=="--cpu-self-test") {std::cout<<"{\"pass\":true,\"cpu_checks\":"<<cpuChecks()<<",\"gpu_commands\":0,\"planned_workspace_bytes\":"<<twoPassPlannedBytes()<<"}\n";return 0;}
      require(argc==3,"usage: oracle METALLIB FRESH_REPORT | --cpu-self-test");require(!std::filesystem::exists(argv[2]),"two-pass report must be fresh");
      require(single("PREFILL_QSA_TWOPASS_ROOT_GPU",0,1)==1,"GPU execution requires Root authorization flag");
      const uint32_t repeats=single("PREFILL_QSA_TWOPASS_PAIRS",8,32);require(repeats>=4&&!(repeats%4),"pairs must be a positive multiple of4");
      const bool packedV=single("PREFILL_QSA_TWOPASS_PACKED_V",0,1);(void)cpuChecks();
      // Standalone arena reservation is fully computed before constructing/allocating the backend.
      require(twoPassPlannedBytes()+bulkExactPlannedBytes()+2ULL*128*24*32*258*4+2ULL*2048*25600*2+64ULL*1024*1024<kStandaloneReservation,
          "standalone preflight2GiB allocation budget exceeded");
      MetalBackend backend(argv[1]);const uint64_t physical=NSProcessInfo.processInfo.physicalMemory;
      const uint64_t hostReserve=splash::engine::EngineMemoryPolicy::hostAvailableReserveBytes(physical);
      require(physical>hostReserve,"insufficient physical memory for original host reserve");
      splash::engine::MemoryGovernor governor(backend,std::min<uint64_t>(8ULL<<30,physical-hostReserve),hostReserve);
      auto reservation=governor.tryReserve(kStandaloneReservation);require(bool(reservation),"real governor denied standalone arena before allocations");
      Input control(backend,2048,20260921),candidate(backend,2048,20260921);
      auto a=allocateQSAState(backend,4096),b=allocateQSAState(backend,4096);StateGuards stateGuardA(backend,a),stateGuardB(backend,b);poisonPadding(a);poisonPadding(b);
      auto wa=allocateQSAWorkspace(backend,128,4096),wb=allocateQSAWorkspace(backend,128,4096);
      auto fa=allocateQSAOnlineMPPWorkspace(backend,128,32),fb=allocateQSAOnlineMPPWorkspace(backend,128,32);
      auto base=allocateBulkExactWorkspace(backend);auto testWorkspace=allocateTwoPassWorkspace(backend,twoPassPlannedBytes());
      GuardedBytes qGuard(backend,25165824),scoreGuard(backend,402653184),rawGuard(backend,50331648),baseRaw(backend,50331648);
      GuardedBytes preparedQGuard(backend,25165824),preparedIndexGuard(backend,2097152),preparedSelectionGuard(backend,4194304);
      testWorkspace.packedQueries=qGuard.view;testWorkspace.scoresAndProbabilities=scoreGuard.view;testWorkspace.rawAttention=rawGuard.view;
      testWorkspace.prepared.queries=preparedQGuard.view;testWorkspace.prepared.indexQueries=preparedIndexGuard.view;testWorkspace.prepared.selectedBlocks=preparedSelectionGuard.view;
      require(backend.memoryStats().peakAllocatedBytes<=kStandaloneReservation,"actual standalone GPU allocation peak exceeded preflight reservation");
      CommandGraph ordinary,baseline,test;addOrdinaryChunkedQSA(backend,ordinary,control.input(),a,wa,fa,0,2048);
      addBulkExactQSA(backend,baseline,control.input(),a,wa,fa,base,0,2048,true);
      addTwoPassQSA(backend,test,candidate.input(),b,wb,fb,testWorkspace,0,2048,false,packedV);
      const auto immutableA=control.hash(),immutableB=candidate.hash();
      std::vector<uint8_t> queries(25165824),indexQueries(2097152);std::vector<uint32_t> selection(2048*512);
      for (uint32_t offset=0;offset<2048;offset+=128) {CommandGraph chunk;
        addOrdinaryQSAChunk(chunk,sliceCoalescedInputs(backend,control.input(),offset,128),a,wa,fa,offset,128);
        (void)backend.submitCommand(chunk.dispatches());
        std::memcpy(queries.data()+uint64_t(offset)*6144*2,wa.queries.contents(),128*6144*2);
        std::memcpy(indexQueries.data()+uint64_t(offset)*512*2,wa.indexQueries.contents(),128*512*2);
        std::memcpy(selection.data()+uint64_t(offset)*512,wa.selectedBlocks.contents(),128*512*4);}
      const std::vector<uint16_t> originalOutput(control.output.values().begin(),control.output.values().end());
      (void)backend.submitCommand(baseline.dispatches());require(compare(control.output.values(),originalOutput).mismatches==0,"SG8 control changed authoritative ordinary output");
      CommandGraph extract;const FlashQSAFastParams extractParams{{2048,0,4096,1024,0,512,0,0,1e-6f,1e7f,0,0},0,0,4,4};
      extract.add("sep21_qsa_twopass_control_raw",{base.partials.partitionStatistics,base.partials.partitionValues,baseRaw.view},extractParams,{2048,24,1},{256,1,1});
      (void)backend.submitCommand(extract.dispatches());
      const auto qkOnly=range(test,0,5);(void)backend.submitCommand(qkOnly.dispatches());
      Certificate scoreCertificate;const auto reference=f64Reference(testWorkspace,b,scoreCertificate);
      {std::ostringstream scoreMetrics;scoreMetrics<<"{\"QK_f64_certificate\":";scoreCertificate.write(scoreMetrics);scoreMetrics<<'}';failureMetrics=scoreMetrics.str();}
      require(!scoreCertificate.failures&&!scoreCertificate.signFailures,"whole-K QK failed preregistered F64 source operand envelope");
      const auto *scoreData=static_cast<const float *>(scoreGuard.view.contents());uint64_t maskedCells=0;
      for (uint32_t kv=0;kv<2;++kv) for (uint32_t flat=0;flat<24576;++flat) for (uint32_t token=0;token<2048;++token) {
        const float s=scoreData[(uint64_t(kv)*24576+flat)*2048+token];if (token>flat/12) {require(s==-INFINITY,"QK failed explicit causal -Inf mask");++maskedCells;}
        else require(std::isfinite(s),"QK valid dense score is nonfinite");}
      (void)backend.submitCommand(test.dispatches());equalState(a,b);
      require(!std::memcmp(queries.data(),testWorkspace.prepared.queries.contents(),queries.size()),"original Q norm/RoPE bytes changed");
      require(!std::memcmp(indexQueries.data(),testWorkspace.prepared.indexQueries.contents(),indexQueries.size()),"original index Q preparation changed");
      require(!std::memcmp(selection.data(),testWorkspace.prepared.selectedBlocks.contents(),selection.size()*4),"original chronology/block/live-tail selection changed");
      const auto *raw=static_cast<const float *>(rawGuard.view.contents()),*rawBase=static_cast<const float *>(baseRaw.view.contents());FloatError rawError;
      for (uint64_t i=0;i<50331648/4;++i) rawError.add(raw[i],rawBase[i]);
      const auto outputError=compare(candidate.output.values(),control.output.values());Certificate rawCertificate,baselineCertificate;Error gatedF64;
      for (const auto &r:reference) for (uint32_t i=0;i<columns.size();++i) {const uint64_t at=(uint64_t(r.head/12)*24576+r.row*12+r.head%12)*256+columns[i];
        rawCertificate.add(raw[at],r.raw[i],kRawAbs,kRawRelative);baselineCertificate.add(rawBase[at],r.raw[i],kRawAbs,kRawRelative);
        const uint64_t out=(uint64_t(r.row)*24+r.head)*256+columns[i];const auto gate=static_cast<const uint16_t *>(candidate.q.contents())[uint64_t(r.row)*12288+r.head*512+256+columns[i]];
        gatedF64.add(candidate.output.values()[out],gated(bf16(float(r.raw[i])),gate));}
      std::ostringstream quality;quality<<std::setprecision(16)<<"{\"raw_source_error\":";rawError.write(quality);quality<<",\"BF16_source_error\":";outputError.write(quality);
      quality<<",\"QK_f64_certificate\":";scoreCertificate.write(quality);quality<<",\"raw_f64_certificate\":";rawCertificate.write(quality);
      quality<<",\"baseline_raw_f64_certificate\":";baselineCertificate.write(quality);quality<<",\"staged_gate_f64_sample_error\":";gatedF64.write(quality);quality<<'}';failureMetrics=quality.str();
      require(!rawError.nonfinite&&!rawError.signFlips&&rawError.l2()<=kSourceRawL2&&rawError.cosine()>=kSourceCosine,"F32 raw source error exceeded preregistration");
      require(!outputError.nonfinite&&outputError.relativeL2()<=kSourceBF16L2&&outputError.cosine()>=kSourceCosine,"BF16 source output exceeded preregistration");
      require(!rawCertificate.failures&&!rawCertificate.signFailures&&!baselineCertificate.failures&&!baselineCertificate.signFailures,"raw F64 producer envelope failed");
      require(!gatedF64.nonfinite&&gatedF64.relativeL2()<=kSourceBF16L2&&gatedF64.cosine()>=kSourceCosine,"staged gate F64 sample exceeded preregistration");
      uint64_t zeroFutureP=0;double maxProbabilitySumError=0;
      const auto *p=static_cast<const float *>(scoreGuard.view.contents());for (uint32_t kv=0;kv<2;++kv) for (uint32_t flat=0;flat<24576;++flat) {
        double sum=0;for (uint32_t token=0;token<2048;++token) {const float probability=p[(uint64_t(kv)*24576+flat)*2048+token];
          require(std::isfinite(probability)&&probability>=0,"F32 P is invalid");if (token>flat/12) {require(probability==0,"future P is not exact zero");++zeroFutureP;}else sum+=probability;}
        maxProbabilitySumError=std::max(maxProbabilitySumError,std::abs(sum-1));}
      require(maxProbabilitySumError<=2e-6,"global F32 P sum exceeded preregistration");
      for (const auto &plane:{b.keys,b.values}) {const auto *cells=static_cast<const uint16_t *>(plane.contents());
        for (uint64_t i=uint64_t(2048)*512;i<uint64_t(4096)*512;++i) require(cells[i]==kSentinel,"physical KV NaN padding changed");}
      // Finite future operands may change, but earlier causal rows stay byte exact.
      const std::vector<uint16_t> first128(candidate.output.values().begin(),candidate.output.values().begin()+128*6144);
      for (const auto &plane:{b.keys,b.values}) {auto *cells=static_cast<uint16_t *>(plane.contents());
        for (uint64_t i=uint64_t(128)*512;i<uint64_t(2048)*512;++i) cells[i]=bf16(-number(cells[i])+1.0f);}
      const auto attentionOnly=range(test,3,uint32_t(test.dispatches().size()));(void)backend.submitCommand(attentionOnly.dispatches());
      require(!std::memcmp(first128.data(),candidate.output.view.contents(),first128.size()*2),"finite future KV changed prior causal output");
      std::memcpy(b.keys.contents(),a.keys.contents(),a.keys.sizeBytes());std::memcpy(b.values.contents(),a.values.contents(),a.values.sizeBytes());
      (void)backend.submitCommand(test.dispatches());
      // Preserve active chronological selection checks and inactive-cell semantics.
      auto *selectedCells=static_cast<uint32_t *>(testWorkspace.prepared.selectedBlocks.contents());const auto originalActive=selectedCells[2047*512+511];
      selectedCells[2047*512+511]=0;*static_cast<uint32_t *>(candidate.diag.contents())=kSticky;
      (void)backend.submitCommand(attentionOnly.dispatches());require(*static_cast<uint32_t *>(candidate.diag.contents())==(kSticky|(1u<<9)),"corrupt active dense selection was silently ignored");
      selectedCells[2047*512+511]=originalActive;const auto originalInactive=selectedCells[511];selectedCells[511]=0x12345678;
      *static_cast<uint32_t *>(candidate.diag.contents())=kSticky;(void)backend.submitCommand(attentionOnly.dispatches());
      require(*static_cast<uint32_t *>(candidate.diag.contents())==kSticky,"inactive selected cell changed dense eligibility");selectedCells[511]=originalInactive;
      (void)backend.submitCommand(test.dispatches());equalState(a,b);
      uint32_t begin=2048;for (uint32_t count:{1u,3u,7u,128u}) {Input x(backend,count,20260921+begin),y(backend,count,20260921+begin);CommandGraph ga,gb;
        addOrdinaryQSAChunk(ga,x.input(),a,wa,fa,begin,count);addOrdinaryQSAChunk(gb,y.input(),b,wb,fb,begin,count);
        (void)backend.submitCommand(ga.dispatches());(void)backend.submitCommand(gb.dispatches());equalState(a,b);
        require(compare(x.output.values(),y.output.values()).mismatches==0,"future sparse append output changed");begin+=count;}
      // All malformed new-kernel calls fail before touching any scratch/output.
      const auto scratchBefore=buffersHash({qGuard.view,scoreGuard.view,rawGuard.view,candidate.output.view});uint32_t shaderRejected=0;
      for (const auto &d:test.dispatches()) if (d.pipelineName.starts_with("sep21_qsa_twopass_")) for (uint32_t fault=0;fault<7;++fault) {
        FlashQSAFastParams bad{};std::memcpy(&bad,d.bytes[0].data,sizeof(bad));auto threads=d.threadsPerThreadgroup;
        switch (fault) {case 0:bad.common.begin=1;break;case 1:bad.common.rows=2047;break;case 2:bad.common.capacity=2047;break;
          case 3:bad.common.capacity=262145;break;case 4:bad.partitions=3;break;case 5:bad.maximum_partitions=4;break;default:threads.x=128;break;}
        std::vector<MetalBuffer> bindings;for (const auto &binding:d.buffers) bindings.push_back(binding.buffer);CommandGraph rejected;
        *static_cast<uint32_t *>(candidate.diag.contents())=kSticky;rejected.add(d.pipelineName,bindings,bad,{1,1,1},threads);(void)backend.submitCommand(rejected.dispatches());
        require(*static_cast<uint32_t *>(candidate.diag.contents())==(kSticky|(1u<<9)),"malformed two-pass kernel did not fail closed");++shaderRejected;}
      require(buffersHash({qGuard.view,scoreGuard.view,rawGuard.view,candidate.output.view})==scratchBefore,"malformed call modified destination planes");
      *static_cast<uint32_t *>(candidate.diag.contents())=kSticky;
      // Specialized rows exercise softmax ties and true exponential underflow.
      auto *probability=static_cast<float *>(scoreGuard.view.contents());const uint64_t anchor=uint64_t(2047*12)*2048;
      for (uint32_t token=0;token<2048;++token) probability[anchor+token]=token? -200.0f:0.0f;
      const auto softmaxOnly=range(test,5,6);(void)backend.submitCommand(softmaxOnly.dispatches());
      require(probability[anchor]==1.0f,"underflow anchor maximum probability changed");for (uint32_t token=1;token<2048;++token) require(probability[anchor+token]==0.0f,"underflow anchor failed exact zero tail");
      for (uint32_t token=0;token<2048;++token) probability[anchor+token]=token<2?0.0f:-200.0f;
      (void)backend.submitCommand(softmaxOnly.dispatches());require(probability[anchor]==.5f&&probability[anchor+1]==.5f,"softmax tie anchor failed");
      // Balanced dyadic V with uniform P has exact-zero whole-K cancellation.
      for (uint32_t token=0;token<2048;++token) {probability[anchor+token]=1.0f/2048;
        for (uint32_t d=0;d<256;++d) static_cast<uint16_t *>(b.values.contents())[uint64_t(token)*512+d]=bf16(token%2?.5f:-.5f);}
      const auto pvOnly=range(test,6,packedV?8:7);(void)backend.submitCommand(pvOnly.dispatches());
      for (uint32_t d=0;d<256;++d) require(static_cast<float *>(rawGuard.view.contents())[uint64_t(2047*12)*256+d]==0,"whole-K cancellation anchor failed exact zero");
      std::memcpy(b.values.contents(),a.values.contents(),a.values.sizeBytes());
      // Original prefix rejects nonfinite fresh V; invalid partial-output parity is not asserted.
      auto *freshV=static_cast<uint16_t *>(candidate.v.contents());const uint32_t nonfiniteAt=2047*512;const auto finiteV=freshV[nonfiniteAt];freshV[nonfiniteAt]=0x7f80;
      *static_cast<uint32_t *>(candidate.diag.contents())=kSticky;(void)backend.submitCommand(test.dispatches());
      require((*static_cast<uint32_t *>(candidate.diag.contents())&(1u<<8))!=0,"nonfinite fresh V did not preserve numeric rejection");
      freshV[nonfiniteAt]=finiteV;*static_cast<uint32_t *>(candidate.diag.contents())=kSticky;
      // Restore valid fullgraphs, then ≥150ms GPU warmwork without CPU tensor access.
      double warmGpu=0;uint32_t warmPairs=0;while (warmGpu<.150||warmPairs<8) {
        const auto x=backend.submitCommand(baseline.dispatches()),y=backend.submitCommand(test.dispatches());warmGpu+=x.gpuSeconds+y.gpuSeconds;++warmPairs;
        require(warmPairs<256,"GPU warm duration failed to progress");}
      Times baselineTimes,candidateTimes;std::vector<std::pair<CommandTiming,CommandTiming>> pairs;
      for (uint32_t i=0;i<repeats;++i) {CommandTiming x,y;
        if (i%4==0||i%4==3) {x=backend.submitCommand(baseline.dispatches());y=backend.submitCommand(test.dispatches());}
        else {y=backend.submitCommand(test.dispatches());x=backend.submitCommand(baseline.dispatches());}
        baselineTimes.add(x);candidateTimes.add(y);pairs.emplace_back(x,y);}
      // Tensor contents, hashes and guards are touched only after every timed pair.
      equalState(a,b);qGuard.check();scoreGuard.check();rawGuard.check();baseRaw.check();preparedQGuard.check();preparedIndexGuard.check();preparedSelectionGuard.check();stateGuardA.check();stateGuardB.check();control.output.check(true);candidate.output.check(true);
      require(backend.memoryStats().peakAllocatedBytes<=kStandaloneReservation,"actual standalone GPU peak exceeded reserved2GiB");
      require(control.hash()==immutableA&&candidate.hash()==immutableB,"source inputs/norm tensors changed");
      require(*static_cast<uint32_t *>(control.diag.contents())==kSticky&&*static_cast<uint32_t *>(candidate.diag.contents())==kSticky,"valid two-pass graph set diagnostics");
      reservation->commit();const auto admission=governor.snapshot();require(!admission.deniedReservations&&admission.hostMeasurementValid,"real governor audit failed");
      std::ostringstream report;report<<std::setprecision(16)<<"{\"schema\":\"splash-prefill-qsa-global-f32P-wholeK-component-v1\",\"execution_complete\":true,\"scope\":\"synthetic_one_layer_component_not_model_generation\",\"numerical_alternative\":true,\"model_quality_qualified\":false,\"packed_V\":"<<(packedV?"true":"false")
        <<",\"workspace_planned_bytes\":"<<twoPassPlannedBytes()<<",\"extra_arena_bytes\":"<<twoPassExtraBytes()<<",\"standalone_preflight_reservation_bytes\":"<<kStandaloneReservation
        <<",\"actual_gpu_allocation_peak_bytes\":"<<backend.memoryStats().peakAllocatedBytes<<",\"real_governor_reservation_before_allocations\":true,\"governor_denied_reservations\":"<<admission.deniedReservations<<",\"host_reserve_bytes\":"<<admission.hostReserveBytes<<",\"final_host_available_bytes\":"<<admission.hostAvailableBytes<<",\"original_prefix_prepared_Q_index_selection_exact\":true,\"five_cache_planes_and_four_future_sparse_appends_exact\":true,\"physical_KV_cache_padding_NaN_preserved\":true,\"finite_future_KV_causal_anchor_pass\":true,\"source_active_selection_rejection_and_inactive_cells_pass\":true,\"underflow_tie_and_exact_cancellation_anchors_pass\":true,\"nonfinite_fresh_V_source_numeric_rejection_preserved\":true,\"invalid_fresh_nonfinite_partial_output_parity_claimed\":false,\"external_guards_pass\":true,\"shader_fail_closed_checks\":"<<shaderRejected
        <<",\"causal_negative_infinity_QK_cells\":"<<maskedCells<<",\"future_probability_exact_zero_cells\":"<<zeroFutureP<<",\"max_probability_sum_error\":"<<maxProbabilitySumError
        <<",\"all_pack_Q_QK_softmax_pack_V_PV_unpack_gate_costs_inclusive\":true,\"warm_gpu_seconds\":"<<warmGpu<<",\"warm_pairs\":"<<warmPairs<<",\"CPU_tensor_access_during_warm_and_pairs\":false,\"paired_order\":\"AB_BA_BA_AB\",\"baseline_dispatches\":"<<baseline.dispatches().size()<<",\"candidate_dispatches\":"<<test.dispatches().size()<<",\"quality\":"<<quality.str()<<",\"baseline_timing\":";
      baselineTimes.write(report);report<<",\"candidate_timing\":";candidateTimes.write(report);report<<",\"gpu_speedup\":"<<median(baselineTimes.gpu)/median(candidateTimes.gpu)<<",\"paired_samples\":[";
      for (uint32_t i=0;i<pairs.size();++i) {if (i) report<<',';report<<"{\"baseline_gpu_seconds\":"<<pairs[i].first.gpuSeconds<<",\"candidate_gpu_seconds\":"<<pairs[i].second.gpuSeconds<<",\"baseline_wall_seconds\":"<<pairs[i].first.wallSeconds<<",\"candidate_wall_seconds\":"<<pairs[i].second.wallSeconds<<'}';}
      report<<"]}";writeReport(argv[2],report.str());std::cout<<report.str()<<'\n';return 0;
    } catch (const std::exception &error) {
      if (argc==3&&!std::filesystem::exists(argv[2])) {try {writeReport(argv[2],std::string("{\"execution_complete\":false,\"error\":")+splash::json::quote(error.what())+",\"failed_stage_metrics\":"+(failureMetrics.empty()?"null":failureMetrics)+"}");}catch (...) {}}
      std::cerr<<"two-pass QSA oracle: "<<error.what()<<'\n';return 1;
    }
  }
}
