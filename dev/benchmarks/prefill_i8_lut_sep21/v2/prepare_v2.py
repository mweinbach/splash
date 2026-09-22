#!/usr/bin/env python3
"""Apply the source-proven guard fix/checkpointing to the private v2 clone only."""
from pathlib import Path

HERE=Path(__file__).resolve().parent

def replace(text,before,after,count=1):
    if text.count(before)!=count:raise RuntimeError(f'v2 source drift: {before!r}')
    return text.replace(before,after)

CHECKPOINT=r'''
std::ofstream gCheckpoints;
std::string gPattern="none";
uint32_t gVariant=0;
template <typename Writer> void checkpoint(const char *event,const Writer &writer) {
  if (!gCheckpoints.is_open()) return;
  std::ostringstream out;out<<std::setprecision(12)<<"{\"event\":"<<splash::json::quote(event)
      <<",\"pattern\":"<<splash::json::quote(gPattern)<<",\"variant\":"<<gVariant;
  writer(out);out<<"}\n";gCheckpoints<<out.str();gCheckpoints.flush();
  require(bool(gCheckpoints),"v2 checkpoint write failed");
}
template <typename Values> void integers(std::ostream &out,const Values &values) {
  out<<'[';bool first=true;for (auto value:values) {if (!first) out<<',';first=false;out<<value;}out<<']';
}
void openCheckpoints(const std::filesystem::path &report) {
  const auto path=report.string()+".checkpoints.jsonl";
  require(!std::filesystem::exists(path),"choose fresh v2 checkpoint path");
  gCheckpoints.open(path);require(bool(gCheckpoints),"cannot create v2 checkpoints");
}
'''

NEGATIVE=r'''
uint32_t negativeIDs(MetalBackend &backend,MetalBuffer input,const std::vector<int64_t> &ids,
    FlashMoEBlockedScratch &scratch,uint32_t rows,std::vector<Guard> &guards) {
  uint32_t checked=0;
  for (uint32_t kind=0;kind<3;++kind) {
    auto malformed=ids;malformed[0]=kind==0 ? -1 :kind==1 ? 512 : malformed[1];
    const auto expected=lut::guard_v2::expected(malformed,rows,10,kSticky);
    const auto routes=upload(backend,malformed,"v2 exact native malformed route IDs");
    const auto diag=guarded(backend,4,guards);*static_cast<uint32_t *>(diag.contents())=kSticky;
    CommandGraph graph;addMoEBlockedPack(graph,input,routes,scratch,diag,rows,FlashMoEBlockedTile::M32N64);
    (void)backend.submitCommand(graph.dispatches());
    const auto actualDiag=*static_cast<const uint32_t *>(diag.contents());
    const auto actualJobs=*static_cast<const uint32_t *>(scratch.buckets.jobCount.contents());
    std::array<uint32_t,512> counts{};std::array<uint32_t,513> offsets{};
    std::memcpy(counts.data(),scratch.buckets.counts.contents(),512*4);
    std::memcpy(offsets.data(),scratch.buckets.offsets.contents(),513*4);
    std::vector<uint32_t> map(ids.size()),inverse(ids.size());
    std::memcpy(map.data(),scratch.buckets.routeMap.contents(),map.size()*4);
    std::memcpy(inverse.data(),scratch.buckets.canonicalToPacked.contents(),inverse.size()*4);
    const bool guardsClean=std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.clean();});
    const bool pass=actualDiag==expected.diagnostic && counts==expected.counts && offsets==expected.offsets &&
        actualJobs==expected.jobs && actualJobs<=moEBucketJobCapacity(rows,10,32) &&
        map==expected.routeMap && inverse==expected.inverse && guardsClean;
    checkpoint("malformed_id_native_bucket",[&](std::ostream &out) {
      out<<",\"kind\":"<<kind<<",\"fixture\":"<<splash::json::quote(kind==0 ? "ID -1" :kind==1 ? "ID512" :"duplicate ID")
          <<",\"native_producer\":\"addMoEBlockedPack/v1 metallib\",\"timing_included\":false"
          <<",\"expected_ids\":";integers(out,malformed);
      out<<",\"actual_diagnostic\":"<<actualDiag<<",\"expected_diagnostic\":"<<expected.diagnostic
          <<",\"actual_hist_total\":"<<std::accumulate(counts.begin(),counts.end(),uint32_t{0})
          <<",\"expected_hist_total\":"<<expected.total()<<",\"actual_excluded_count\":"<<ids.size()-offsets[512]
          <<",\"expected_excluded_count\":"<<expected.invalidRoutes.size()
          <<",\"expected_duplicate_count\":"<<expected.duplicateRoutes.size()
          <<",\"actual_job_count\":"<<actualJobs<<",\"expected_job_count\":"<<expected.jobs
          <<",\"actual_counts\":";integers(out,counts);out<<",\"expected_counts\":";integers(out,expected.counts);
      out<<",\"actual_offsets\":";integers(out,offsets);out<<",\"expected_offsets\":";integers(out,expected.offsets);
      out<<",\"route_map_exact\":"<<(map==expected.routeMap ? "true" :"false")
          <<",\"canonical_inverse_exact\":"<<(inverse==expected.inverse ? "true" :"false")
          <<",\"guards_clean\":"<<(guardsClean ? "true" :"false")<<",\"pass\":"<<(pass ? "true" :"false");
    });
    require(actualDiag==expected.diagnostic,"v2 malformed ID sticky diagnostic differs");
    require(actualJobs==expected.jobs && actualJobs<=moEBucketJobCapacity(rows,10,32),"v2 malformed ID job count differs");
    require(counts==expected.counts,"v2 exact native malformed histogram differs");
    require(offsets==expected.offsets,"v2 exact native malformed offsets differ");
    require(map==expected.routeMap && inverse==expected.inverse,"v2 exact native malformed maps differ");
    require(guardsClean,"v2 malformed ID canary changed");++checked;
  }
  return checked;
}
'''

GUARDS=r'''
      if (argc==5 && std::string_view(argv[1])=="--guards") {
        const uint32_t rows=numeric(argv[3],2048,2048);const std::filesystem::path report(argv[4]);
        require(!std::filesystem::exists(report),"choose fresh v2 guard report");openCheckpoints(report);
        MetalBackend backend(argv[2]);std::vector<Guard> guards;
        const auto hidden=normalizedHidden(rows);const auto input=upload(backend,hidden,"guard-only row RMS input");
        auto scratch=allocateMoEBlockedScratch(backend,rows);guardScratch(backend,scratch,guards);
        std::vector<uint32_t> hot(512);std::iota(hot.begin(),hot.end(),0u);uint32_t checks=0;
        for (const char *pattern:{"spread-all","hit-concentrated"}) {
          gPattern=pattern;gVariant=0;const auto ids=patternIDs(rows,hot,pattern);
          checks+=negativeIDs(backend,input,ids,scratch,rows,guards);
        }
        std::ofstream out(report);require(bool(out),"cannot create v2 guard report");
        out<<"{\"schema\":\"prefill-i8-lut-native-guards-v2\",\"pass\":true,\"gpu_executed\":true"
            <<",\"guard_checks\":"<<checks<<",\"coefficient_payload_loaded\":false"
            <<",\"candidate_fidelity_measured\":false,\"timing_attempted\":false}\n";out.flush();
        require(bool(out),"v2 guard report write failed");backend.stop();return 0;
      }
'''

def main():
    source=(HERE/'generate.py').read_text()
    source=replace(source,'namespace lut = splash::bench::i8_lut;','namespace lut = splash::bench::i8_lut;\n'+CHECKPOINT)
    start=source.index('uint32_t negativeIDs(');end=source.index('\nbool runCase(',start)
    source=source[:start]+NEGATIVE+source[end:]
    source=replace(source,'  std::vector<uint32_t> hot(512);for (uint32_t i=0;i<512;++i) hot[i]=i;',
        '  gPattern=pattern;gVariant=0;\n  std::vector<uint32_t> hot(512);for (uint32_t i=0;i<512;++i) hot[i]=i;')
    source=replace(source,'    const auto &v=lut::variants[index];','    gVariant=index+1;\n    const auto &v=lut::variants[index];')
    point='    const bool eligible=initialMatched.pass() && initial[1].chainPass() && (!sameBestLoop || initial[1].pass());'
    source=replace(source,point,point+r'''
    checkpoint("candidate_initial_fidelity",[&](std::ostream &out) {
      out<<",\"suffix\":"<<splash::json::quote(v.suffix)<<",\"eligible_before_timing\":"<<(eligible ? "true" :"false")
          <<",\"timing_attempted\":false,\"against_whole_sg4\":";initial[0].write(out);
      out<<",\"against_current_best_sg2k128\":";initial[1].write(out);
      out<<",\"against_matched_uncompressed\":";initialMatched.write(out);
    });
    const uint32_t producerNegativeChecks=negativeProducers(backend,candidateAudit,audits[2],scratch[0],rows,guards);
    const uint32_t idNegativeChecks=negativeIDs(backend,input,ids,scratch[2],rows,guards);
    (void)backend.submitCommand(controls[0]);(void)backend.submitCommand(controls[1]);
    (void)backend.submitCommand(plan.commands);healthy();
    checkpoint("malformed_checks_before_timing_pass",[&](std::ostream &out) {
      out<<",\"producer_checks\":"<<producerNegativeChecks<<",\"id_checks\":"<<idNegativeChecks
          <<",\"clean_states_restored\":true,\"timing_attempted\":false";
    });
''')
    source=replace(source,'    const uint32_t producerNegativeChecks=negativeProducers(backend,candidateAudit,audits[2],scratch[0],rows,guards);\n    const uint32_t idNegativeChecks=negativeIDs(backend,input,ids,scratch[2],rows,guards);\n    (void)backend.submitCommand(plan.commands);healthy();',r'''
    checkpoint("candidate_final_fidelity_and_timings",[&](std::ostream &out) {
      out<<",\"pass\":"<<(pass ? "true" :"false")<<",\"timing_attempted\":"<<(eligible ? "true" :"false")
          <<",\"against_whole_sg4\":";final[0].write(out);
      out<<",\"against_current_best_sg2k128\":";final[1].write(out);
      out<<",\"against_matched_uncompressed\":";finalMatched.write(out);
      out<<",\"timing_scopes\":[";for (uint32_t i=0;i<measured.size();++i) {if (i) out<<',';measured[i].write(out);}out<<']';
    });
    (void)backend.submitCommand(plan.commands);healthy();''')
    source=replace(source,'        cpuSelfTest();lut::cpuSelfTest();','        cpuSelfTest();lut::cpuSelfTest();lut::guard_v2::cpuSelfTest();')
    source=replace(source,'      require(argc==6 && std::string_view(argv[1])=="--gpu",',GUARDS+'      require(argc==6 && std::string_view(argv[1])=="--gpu",')
    source=replace(source,'      const auto metadata=lut::Metadata::load(argv[3]);',
        '      openCheckpoints(argv[5]);\n      const auto metadata=lut::Metadata::load(argv[3]);')
    source=replace(source,'prefill-i8-lut-sep21-strict-bounded-v1','prefill-i8-lut-sep21-strict-bounded-v2')
    source=replace(source,'    } catch (const std::exception &e) {std::cerr<<e.what()<<\'\\n\';return 1;}',
        '''    } catch (const std::exception &e) {
      checkpoint("failure",[&](std::ostream &out){out<<",\\\"failure\\\":"<<splash::json::quote(e.what());});
      std::cerr<<e.what()<<'\\n';return 1;
    }''')
    source=source.replace('dev/benchmarks/prefill_i8_lut_sep21/cache.hpp','dev/benchmarks/prefill_i8_lut_sep21/v2/cache.hpp')
    point='#include "dev/benchmarks/prefill_i8_lut_sep21/v2/cache.hpp"'
    source=replace(source,point,point+'\\n#include "dev/benchmarks/prefill_i8_lut_sep21/v2/guard_reference.hpp"')
    (HERE/'generate.py').write_text(source)

if __name__=='__main__':main()
