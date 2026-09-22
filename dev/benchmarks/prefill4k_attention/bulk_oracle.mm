#define main prefill4k_synthetic_oracle_main
#include "oracle.mm"
#undef main
#include "bulk.hpp"
#include <tuple>
namespace {
using namespace splash::flash::prefill4k;
FlashTensor tensor(MetalBackend &b,uint32_t width,bool f32,uint64_t seed) {
  FlashTensor t;t.dtype=f32?FlashDType::F32:FlashDType::BF16;t.shape={width};t.logicalBytes=uint64_t(width)*(f32?4:2);
  t.buffer=b.allocateBuffer(t.logicalBytes,BufferStorage::Shared,"bulk norm");
  for (uint32_t i=0;i<width;++i) {
    const float v=.2f+seeded(i,seed,.1f);
    if (f32) static_cast<float *>(t.buffer.contents())[i]=v;
    else static_cast<uint16_t *>(t.buffer.contents())[i]=bf16(v);
  }return t;
}
struct Inputs {
  MetalBuffer q,k,v,index,diag;FlashTensor qn,kn,iqn,ikn;Guarded output;uint32_t rows;
  Inputs(MetalBackend &b,uint32_t n,uint64_t seed):qn(tensor(b,256,true,111)),kn(tensor(b,256,false,112)),
      iqn(tensor(b,128,false,113)),ikn(tensor(b,128,true,114)),output(b,uint64_t(n)*6144),rows(n) {
    auto fill=[&](uint32_t width,uint64_t salt,float scale) {
      auto x=b.allocateBuffer(uint64_t(n)*width*2,BufferStorage::Shared,"bulk projected input");
      auto *p=static_cast<uint16_t *>(x.contents());for (uint64_t i=0;i<uint64_t(n)*width;++i) p[i]=bf16(seeded(i,seed+salt,scale));return x;
    };
    q=fill(12288,1,2);k=fill(512,2,2);v=fill(512,3,.75);index=fill(640,4,1.5);
    diag=b.allocateBuffer(4,BufferStorage::Shared,"bulk diagnostic");*static_cast<uint32_t *>(diag.contents())=kSticky;
  }
  FlashQSAFastInputs input() {
    return {q,k,v,index,&qn,&kn,&iqn,&ikn,output.view,diag,{},NormConvention::OnePlusWeight,
            NormConvention::OnePlusWeight,NormConvention::OnePlusWeight,NormConvention::OnePlusWeight,1e-6,1e7};
  }
  std::string hash() const {SHA256 h;for (const auto &x:{q,k,v,index,qn.buffer,kn.buffer,iqn.buffer,ikn.buffer})h.add(x);return h.finish();}
};
std::array<MetalBuffer,5> planes(const FlashQSAState &s) {return {s.keys,s.values,s.rawIndexKeys,s.pooledKeys,s.indexPositions};}
void clearState(const FlashQSAState &s) {for (const auto &x:planes(s))std::memset(x.contents(),0,x.sizeBytes());}
void stateEqual(const FlashQSAState &a,const FlashQSAState &b) {
  auto aa=planes(a),bb=planes(b);
  for (uint32_t i=0;i<5;++i) require(aa[i].sizeBytes()==bb[i].sizeBytes()&&!std::memcmp(aa[i].contents(),bb[i].contents(),aa[i].sizeBytes()),"Bulk QSA persistent cache bytes changed");
}
}
int main(int argc,char **argv) {
  @autoreleasepool {
    std::string failureMetrics;
    try {
      if (argc==2&&std::string(argv[1])=="--cpu-self-test") {
        require(bulkExactPlannedBytes()==234356736,"Bulk scratch byte plan differs");
        require(denseCoalescedPlannedBytes(2048)==31457280,"Direct scratch must be30MiB");
        std::cout << "{\"pass\":true,\"gpu_commands\":0,\"exact_planned_bytes\":" << bulkExactPlannedBytes() << "}\n";return 0;
      }
      require(argc==3,"usage: bulk-oracle METALLIB FRESH_REPORT | --cpu-self-test");
      require(!std::filesystem::exists(argv[2]),"Bulk report must be fresh");
      const bool bf16PV=single("PREFILL4K_BULK_BF16PV",0,1);
      const bool sg8=single("PREFILL4K_BULK_SG8",0,1);
      const bool direct=single("PREFILL4K_BULK_DIRECT",0,1)||bf16PV;
      const uint32_t repeats=single("PREFILL4K_BULK_REPEATS",5,100);require(repeats,"Bulk repeats must be positive");
      MetalBackend backend(argv[1]);Inputs control(backend,2048,20260921),candidate(backend,2048,20260921);
      auto a=allocateQSAState(backend,4096),b=allocateQSAState(backend,4096);clearState(a);clearState(b);
      auto wa=allocateQSAWorkspace(backend,128,4096),wb=allocateQSAWorkspace(backend,128,4096);
      auto fa=allocateQSAOnlineMPPWorkspace(backend,128,32),fb=allocateQSAOnlineMPPWorkspace(backend,128,32);
      DenseCoalescedWorkspace prepared;BulkExactWorkspace bulk;
      if (direct) prepared=allocateDenseCoalescedWorkspace(backend,2048);else bulk=allocateBulkExactWorkspace(backend);
      auto &preparedCandidate=direct?prepared:bulk.prepared;
      CommandGraph ordinary,test;addOrdinaryChunkedQSA(backend,ordinary,control.input(),a,wa,fa,0,2048);
      if (direct) addBulkDirectQSA(backend,test,candidate.input(),b,wb,fb,prepared,0,2048,bf16PV);
      else addBulkExactQSA(backend,test,candidate.input(),b,wb,fb,bulk,0,2048,sg8);
      const auto sourceA=control.hash(),sourceB=candidate.hash();
      std::vector<uint8_t> expectedQueries(uint64_t(2048)*6144*2),expectedIndices(uint64_t(2048)*512*2);
      std::vector<uint32_t> expectedSelected(uint64_t(2048)*512);
      std::vector<float> expectedStats(uint64_t(2048)*24*4*2),expectedNums(uint64_t(2048)*24*4*256);
      for (uint32_t offset=0;offset<2048;offset+=128) {
        CommandGraph chunk;addOrdinaryQSAChunk(chunk,sliceCoalescedInputs(backend,control.input(),offset,128),a,wa,fa,offset,128);
        (void)backend.submitCommand(chunk.dispatches());
        std::memcpy(expectedQueries.data()+uint64_t(offset)*6144*2,wa.queries.contents(),128*6144*2);
        std::memcpy(expectedIndices.data()+uint64_t(offset)*512*2,wa.indexQueries.contents(),128*512*2);
        std::memcpy(expectedSelected.data()+uint64_t(offset)*512,wa.selectedBlocks.contents(),128*512*4);
        const uint32_t parts=offset?4:1;
        const auto *stats=static_cast<const float *>(fa.partitionStatistics.contents()),*nums=static_cast<const float *>(fa.partitionValues.contents());
        for (uint32_t row=0;row<128;++row)for (uint32_t h=0;h<24;++h)for (uint32_t p=0;p<parts;++p) {
          const uint64_t from=(uint64_t(row)*24+h)*32+p,to=((uint64_t(offset)+row)*24+h)*4+p;
          std::copy_n(stats+from*2,2,expectedStats.data()+to*2);std::copy_n(nums+from*256,256,expectedNums.data()+to*256);
        }
      }
      (void)backend.submitCommand(test.dispatches());stateEqual(a,b);
      require(!std::memcmp(expectedQueries.data(),preparedCandidate.queries.contents(),expectedQueries.size()),"Bulk prepared Q bytes changed");
      require(!std::memcmp(expectedIndices.data(),preparedCandidate.indexQueries.contents(),expectedIndices.size()),"Bulk prepared index Q bytes changed");
      require(!std::memcmp(expectedSelected.data(),preparedCandidate.selectedBlocks.contents(),expectedSelected.size()*4),"Bulk dense chronological selection changed");
      bool partialsExact=true;
      if (!direct) {
        const auto *stats=static_cast<const float *>(bulk.partials.partitionStatistics.contents()),*nums=static_cast<const float *>(bulk.partials.partitionValues.contents());
        for (uint32_t row=0;row<2048;++row)for (uint32_t h=0;h<24;++h)for (uint32_t p=0;p<(row<128?1u:4u);++p) {
          const uint64_t at=(uint64_t(row)*24+h)*4+p;
          partialsExact &= !std::memcmp(stats+at*2,expectedStats.data()+at*2,2*4)&&!std::memcmp(nums+at*256,expectedNums.data()+at*256,256*4);
        }
        require(partialsExact,"Exact bulk F32 attention partials changed");
      }
      auto error=compare(candidate.output.values(),control.output.values());
      std::ostringstream diagnostic;
      diagnostic << std::setprecision(12) << "{\"stage\":\"BF16_output\",\"prepared_cache_partials_checked\":true,\"output_error\":";error.write(diagnostic);
      diagnostic << ",\"mismatches_by_128_window\":[";
      std::array<uint64_t,16> mismatches{};
      for (uint64_t i=0;i<uint64_t(2048)*6144;++i)
        if (candidate.output.values()[i]!=control.output.values()[i]) ++mismatches[i/6144/128];
      for (uint32_t w=0;w<16;++w) {if(w)diagnostic << ',';diagnostic << mismatches[w];}
      diagnostic << "],\"first_changed_cells\":[";uint32_t changes=0;
      for (uint64_t i=0;i<uint64_t(2048)*6144&&changes<32;++i)
        if (candidate.output.values()[i]!=control.output.values()[i]) {
          if(changes++)diagnostic << ',';
          diagnostic << "{\"row\":" << i/6144 << ",\"head\":" << i%6144/256
              << ",\"dimension\":" << i%256 << ",\"baseline_bits\":" << control.output.values()[i]
              << ",\"candidate_bits\":" << candidate.output.values()[i] << '}';
        }
      diagnostic << "]}";failureMetrics=diagnostic.str();
      if (!direct)require(error.mismatches==0,"Exact bulk BF16 output changed");
      control.output.check(true);candidate.output.check(true);
      require(*static_cast<uint32_t *>(control.diag.contents())==kSticky&&*static_cast<uint32_t *>(candidate.diag.contents())==kSticky,"Bulk route set diagnostics");
      uint32_t begin=2048;
      for (uint32_t count:{1u,3u,7u,128u}) {
        Inputs x(backend,count,20260921+begin),y(backend,count,20260921+begin);CommandGraph ga,gb;
        addOrdinaryQSAChunk(ga,x.input(),a,wa,fa,begin,count);addOrdinaryQSAChunk(gb,y.input(),b,wb,fb,begin,count);
        (void)backend.submitCommand(ga.dispatches());(void)backend.submitCommand(gb.dispatches());stateEqual(a,b);
        require(compare(x.output.values(),y.output.values()).mismatches==0,"Bulk cache changed subsequent sparse append output");begin+=count;
      }
      Times oldTimes,newTimes;
      for (uint32_t i=0;i<repeats+2;++i) {
        clearState(a);clearState(b);CommandTiming x,y;
        if (i%2) {y=backend.submitCommand(test.dispatches());x=backend.submitCommand(ordinary.dispatches());}
        else {x=backend.submitCommand(ordinary.dispatches());y=backend.submitCommand(test.dispatches());}
        stateEqual(a,b);if(i>=2){oldTimes.add(x);newTimes.add(y);}
      }
      require(control.hash()==sourceA&&candidate.hash()==sourceB,"Bulk route modified projected inputs/norms");
      std::ostringstream report;report << std::setprecision(12) << "{\"execution_complete\":true,\"scope\":\"synthetic_full_QSA_not_model_generation\",\"numerical_alternative\":" << (direct?"true":"false") << ",\"temporal_sg8\":" << (sg8?"true":"false") << ",\"tile_local_bf16_probability\":" << (bf16PV?"true":"false") << ",\"ordinary_dispatches\":" << ordinary.dispatches().size() << ",\"bulk_dispatches\":" << test.dispatches().size() << ",\"prepared_queries_index_selection_exact\":true,\"all_five_cache_planes_and_four_future_sparse_appends_exact\":true,\"projected_inputs_norms_unchanged\":true,\"f32_partials_exact\":" << ((!direct&&partialsExact)?"true":"false") << ",\"output_error\":";error.write(report);
      report << ",\"baseline_timing\":";oldTimes.write(report);report << ",\"candidate_timing\":";newTimes.write(report);report << ",\"gpu_speedup\":" << median(oldTimes.gpu)/median(newTimes.gpu) << '}';writeReport(argv[2],report.str());std::cout << report.str() << '\n';return 0;
    } catch(const std::exception &error) {
      if (argc==3 && !std::filesystem::exists(argv[2])) {
        try {writeReport(argv[2],std::string("{\"execution_complete\":false,\"error\":")+
          splash::json::quote(error.what())+",\"failed_stage_metrics\":"+
          (failureMetrics.empty()?"null":failureMetrics)+"}");}catch(...){}
      }
      std::cerr << "bulk QSA oracle: " << error.what() << '\n';return 1;
    }
  }
}
