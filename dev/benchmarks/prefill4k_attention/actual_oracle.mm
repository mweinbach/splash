#define main prefill4k_synthetic_oracle_main
#include "oracle.mm"
#undef main

namespace {
uint32_t integer(id value,const char *name) {
  require([value isKindOfClass:[NSNumber class]] && CFGetTypeID((__bridge CFTypeRef)value)!=CFBooleanGetTypeID(),name);
  const double d=[value doubleValue];require(std::isfinite(d)&&d>=0&&d<=262144&&std::floor(d)==d,name);
  return uint32_t(d);
}
std::vector<uint32_t> shape(NSDictionary *entry) {
  id data=entry[@"shape"];require([data isKindOfClass:[NSArray class]],"Actual array has no shape");
  std::vector<uint32_t> result;for (id v in static_cast<NSArray *>(data)) result.push_back(integer(v,"Actual shape must be integer"));
  return result;
}
void load(NSDictionary *arrays,NSString *name,const std::filesystem::path &folder,
          const std::vector<uint32_t> &dimensions,MetalBuffer destination) {
  id entry=arrays[name];require([entry isKindOfClass:[NSDictionary class]],"Actual operand missing");
  NSDictionary *item=entry;
  require([item[@"dtype"] isEqualToString:@"<u2"] && shape(item)==dimensions,"Actual BF16 dtype/shape mismatch");
  require([item[@"file"] isKindOfClass:[NSString class]],"Actual array file missing");
  const auto file=folder/[item[@"file"] UTF8String];
  NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:file.c_str()]];
  uint64_t bytes=2;for (auto dim:dimensions) bytes*=dim;
  require(data && [data length]==bytes && destination.sizeBytes()>=bytes,"Actual BF16 operand extent mismatch");
  SHA256 hash;hash.add([data bytes],bytes);
  if (item[@"sha256"]) require(hash.finish()==std::string([item[@"sha256"] UTF8String]),"Actual array checksum mismatch");
  std::memcpy(destination.contents(),[data bytes],bytes);
}
void dump(const std::filesystem::path &folder,const char *name,const MetalBuffer &buffer) {
  std::ofstream output(folder/name,std::ios::binary);require(bool(output),"Cannot write actual result");
  output.write(static_cast<const char *>(buffer.contents()),std::streamsize(buffer.sizeBytes()));require(bool(output),"Actual result write incomplete");
}
}
int main(int argc,char **argv) {
  @autoreleasepool {
    try {
      if (argc==2 && std::string(argv[1])=="--help") {
        std::cout << "usage: actual-oracle METALLIB CAPTURE_MANIFEST FRESH_OUTPUT_DIRECTORY\n"
                     "PREFILL4K_ACTUAL_REPEATS=7; candidate flags match synthetic oracle. No model load.\n";return 0;
      }
      require(argc==4,"usage: actual-oracle METALLIB CAPTURE_MANIFEST FRESH_OUTPUT_DIRECTORY");
      const auto manifestPath=std::filesystem::absolute(argv[2]),folder=manifestPath.parent_path(),out=std::filesystem::absolute(argv[3]);
      require(!std::filesystem::exists(out),"Actual output directory must be fresh");
      NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:manifestPath.c_str()]];
      NSError *error=nil;id document=data?[NSJSONSerialization JSONObjectWithData:data options:0 error:&error]:nil;
      require(document && !error && [document isKindOfClass:[NSDictionary class]],"Cannot parse actual capture manifest");
      NSDictionary *metadata=document;NSDictionary *arrays=metadata[@"arrays"];
      require([arrays isKindOfClass:[NSDictionary class]],"Actual capture arrays missing");
      const auto qShape=shape(arrays[@"queries_bf16"]),kShape=shape(arrays[@"keys_bf16"]);
      require(qShape.size()==3&&qShape[1]==24&&qShape[2]==256&&qShape[0]>=32&&qShape[0]<=128,"Actual Q geometry unsupported");
      require(kShape.size()==3&&kShape[1]==2&&kShape[2]==256,"Actual K geometry unsupported");
      const uint32_t rows=qShape[0],begin=integer(metadata[@"query_offset"],"Actual query offset invalid"),capacity=kShape[0];
      require(flash_qsa_row_tiles_geometry(begin,rows,4)&&begin+rows<=capacity,"Actual capture must be dense current M32 geometry");
      const bool queryReuse=single("PREFILL4K_ATTENTION_QUERY_REUSE",0,1),wide=single("PREFILL4K_ATTENTION_VALUES_N128",0,1);
      const bool strict=single("PREFILL4K_ATTENTION_REGISTER_PV_STRICT",0,1);
      const bool registers=single("PREFILL4K_ATTENTION_REGISTER_PV",0,1)||strict;
      const bool relaxed=single("PREFILL4K_ATTENTION_RELAXED_PV",0,1);
      const bool sg8=single("PREFILL4K_ATTENTION_SG8",0,1);
      const uint32_t repeats=single("PREFILL4K_ACTUAL_REPEATS",7,100);require(repeats,"Actual repeats must be positive");
      MetalBackend backend(argv[1]);auto state=allocateQSAState(backend,capacity);auto workspace=allocateQSAWorkspace(backend,rows,capacity);
      auto fast=allocateQSAOnlineMPPWorkspace(backend,rows,32);
      for (const auto &buffer:{state.rawIndexKeys,state.pooledKeys,state.indexPositions,workspace.indexQueries})
        std::memset(buffer.contents(),0,buffer.sizeBytes());
      load(arrays,@"queries_bf16",folder,qShape,workspace.queries);load(arrays,@"keys_bf16",folder,kShape,state.keys);
      load(arrays,@"values_bf16",folder,kShape,state.values);
      auto gates=backend.allocateBuffer(uint64_t(rows)*6144*2,BufferStorage::Shared,"actual gate bits");load(arrays,@"gates_bf16",folder,qShape,gates);
      auto projection=backend.allocateBuffer(uint64_t(rows)*12288*2,BufferStorage::Shared,"actual gate interleaving");
      std::memset(projection.contents(),0,projection.sizeBytes());const auto *gateBits=static_cast<const uint16_t *>(gates.contents());
      auto *projectionBits=static_cast<uint16_t *>(projection.contents());
      for (uint32_t row=0;row<rows;++row) for (uint32_t h=0;h<24;++h)
        std::copy_n(gateBits+(uint64_t(row)*24+h)*256,256,projectionBits+(uint64_t(row)*24+h)*512+256);
      auto *selected=static_cast<uint32_t *>(workspace.selectedBlocks.contents());
      for (uint32_t row=0;row<rows;++row) for (uint32_t b=0;b<512;++b)
        selected[uint64_t(row)*512+b]=b<(begin+row+1)/4?b:UINT32_MAX;
      auto diagnostics=backend.allocateBuffer(4,BufferStorage::Shared,"actual diagnostics");auto *status=static_cast<uint32_t *>(diagnostics.contents());*status=kSticky;
      Guarded baselineOutput(backend,uint64_t(rows)*6144),candidateOutput(backend,uint64_t(rows)*6144),rawRounded(backend,uint64_t(rows)*6144);
      auto raw=backend.allocateBuffer(uint64_t(rows)*6144*4,BufferStorage::Shared,"actual raw F32 attention");
      CommandGraph baseline,candidate,debug;
      addTemporalCandidate(baseline,projection,state,workspace,fast,baselineOutput.view,diagnostics,begin,rows,4,32);
      addTemporalCandidate(candidate,projection,state,workspace,fast,candidateOutput.view,diagnostics,begin,rows,4,32,queryReuse,wide,registers,strict,relaxed,sg8);
      FlashQSAFastParams params{{rows,begin,capacity,(capacity+3)/4,begin/4,(begin+rows)/4-begin/4,0,0,1e-6f,1e7f,0,0},0,0,4,32};
      debug.add("prefill4k_qsa_attention_debug",{fast.partitionStatistics,fast.partitionValues,raw,rawRounded.view,diagnostics},params,{rows,24,1},{256,1,1});
      const auto sourceHash=inputHash(state,workspace,projection);
      std::filesystem::create_directories(out);
      const auto snap=[](const MetalBuffer &b) {const auto *p=static_cast<const uint8_t *>(b.contents());return std::vector<uint8_t>(p,p+b.sizeBytes());};
      (void)backend.submitCommand(baseline.dispatches());const auto stats=snap(fast.partitionStatistics),numerators=snap(fast.partitionValues);
      for (const auto &[name,graph,output] : {std::tuple{"baseline",&baseline,baselineOutput.view},std::tuple{"candidate",&candidate,candidateOutput.view}}) {
        (void)backend.submitCommand(graph->dispatches());(void)backend.submitCommand(debug.dispatches());
        require(*status==kSticky,"Actual route set diagnostics");const auto target=out/name;std::filesystem::create_directory(target);
        dump(target,"attention-f32.bin",raw);dump(target,"attention-bf16.bin",rawRounded.view);dump(target,"output-bf16.bin",output);
        std::ofstream result(target/"manifest.json");result << "{\"metadata\":{\"actual_activations\":true,\"capture_manifest\":" << splash::json::quote(manifestPath.string()) << "},\"arrays\":{";
        bool first=true;
        for (const auto &[array,file,dtype] : {std::tuple{"candidate_attention_f32","attention-f32.bin","<f4"},std::tuple{"candidate_attention_bf16","attention-bf16.bin","<u2"},std::tuple{"candidate_output_bf16","output-bf16.bin","<u2"}}) {
          if (!first) result << ',';first=false;
          result << splash::json::quote(array) << ":{\"file\":" << splash::json::quote(file) << ",\"dtype\":" << splash::json::quote(dtype) << ",\"shape\":[" << rows << ",24,256]}";
        }
        result << "}}\n";require(bool(result),"Cannot write actual candidate manifest");
      }
      const bool statisticsExact=snap(fast.partitionStatistics)==stats,numeratorsExact=snap(fast.partitionValues)==numerators;
      if (!sg8) require(statisticsExact,"Actual variant changed score/softmax statistics");
      baselineOutput.check(true);candidateOutput.check(true);rawRounded.check(true);
      const auto comparison=compare(candidateOutput.values(),baselineOutput.values());
      Times baseTimes,candidateTimes;
      for (uint32_t i=0;i<repeats+2;++i) {
        CommandTiming a,b;
        if (i%2) {b=backend.submitCommand(candidate.dispatches());a=backend.submitCommand(baseline.dispatches());}
        else {a=backend.submitCommand(baseline.dispatches());b=backend.submitCommand(candidate.dispatches());}
        if (i>=2) {baseTimes.add(a);candidateTimes.add(b);}
      }
      require(*status==kSticky,"Actual timing route set diagnostics");
      require(inputHash(state,workspace,projection)==sourceHash,"Actual variant modified immutable inputs/caches");
      std::ostringstream report;report << std::setprecision(12) << "{\"execution_complete\":true,\"actual_activations\":true,\"scope\":\"attention_only_not_model_generation\",\"maximum_partitions\":32,\"f32_statistics_exact\":" << (statisticsExact?"true":"false") << ",\"f32_numerators_exact\":" << (numeratorsExact?"true":"false") << ",\"baseline_timing\":";baseTimes.write(report);
      report << ",\"candidate_timing\":";candidateTimes.write(report);report << ",\"gpu_speedup\":" << median(baseTimes.gpu)/median(candidateTimes.gpu) << ",\"output_error\":";comparison.write(report);report << '}';
      writeReport(out/"report.json",report.str());std::cout << report.str() << '\n';return 0;
    } catch (const std::exception &error) {std::cerr << "actual QSA oracle: " << error.what() << '\n';return 1;}
  }
}
