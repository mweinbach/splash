// Private source-certified Q27 projection screen. Only Root runs GPU jobs.
#include "metal/MetalBackend.hpp"
#include "engine/Json.hpp"
#include "metal/abi/Linear.h"
#include "FlashFloatBoundaryAudit.hpp"
#include "prefill4k_q27_params.h"
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace splash::metal;
using namespace splash::flash::benchmark;
void require(bool value,const char *message) { if(!value)throw std::runtime_error(message); }
constexpr uint32_t guardWords=32, sticky=0x40000000;
constexpr uint16_t marker16=0x7fc1;
constexpr uint32_t marker32=0x7fc12345;
uint32_t extent(uint32_t n,uint32_t tile) { return (n+tile-1)/tile*tile; }
uint64_t randomWord(uint64_t x) {
  x+=0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
  x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31);
}
uint32_t envPositive(const char *key,uint32_t fallback,uint32_t maximum) {
  const auto raw=std::getenv(key); if(!raw)return fallback;
  size_t used=0;require(*raw&&*raw!='-',"invalid positive environment setting");
  const auto result=std::stoul(raw,&used);
  require(used==std::strlen(raw)&&result&&result<=maximum,"invalid positive environment setting");
  return uint32_t(result);
}
std::string digest(const void *data,size_t bytes) {
  require(bytes<=UINT32_MAX,"digest extent exceeds single SHA256 call");
  std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> value{};
  CC_SHA256(data,CC_LONG(bytes),value.data());
  std::ostringstream out;for(auto x:value)out<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(x);
  return out.str();
}
struct Fixture {
  uint32_t n=0,k=0;
  std::vector<uint8_t> packed;
  std::vector<uint16_t> scales,biases,bf16Coefficients;
  std::vector<uint32_t> f32Coefficients;
  uint64_t sourceChecks=0;
  std::string packedSHA,f32SHA,bf16SHA;
  uint64_t parameter(uint32_t nIndex,uint32_t kIndex)const {
    return (uint64_t(nIndex/256)*(k/64)+kIndex/64)*256+nIndex%256;
  }
  uint32_t code(uint32_t nIndex,uint32_t kIndex)const {
    const auto value=packed[parameter(nIndex,kIndex)*32+(kIndex%64)/2];
    return (value>>(4*(kIndex&1)))&15;
  }
  double exact(uint32_t nIndex,uint32_t kIndex)const {
    const auto p=parameter(nIndex,kIndex);
    return double(code(nIndex,kIndex))*number(scales[p])+number(biases[p]);
  }
};
Fixture loadFixture(const char *path) {
  require(std::endian::native==std::endian::little,"fixture requires little endian host");
  std::ifstream stream(path,std::ios::binary);require(bool(stream),"cannot open source fixture");
  const auto read=[&](void *data,size_t bytes) {
    stream.read(static_cast<char *>(data),std::streamsize(bytes));
    require(size_t(stream.gcount())==bytes,"truncated source fixture");
  };
  std::array<char,8> magic{};std::array<uint32_t,10> fields{};
  read(magic.data(),magic.size());read(fields.data(),fields.size()*4);
  require(std::string(magic.data(),8)=="Q27A0001"&&fields[0]&&fields[0]<=32768&&
      fields[1]&&fields[1]<=32768&&fields[0]%256==0&&fields[1]%64==0&&
      fields[2]==64&&fields[3]==256&&!fields[9],"invalid fixture header");
  Fixture result;result.n=fields[0];result.k=fields[1];
  const uint64_t elements=uint64_t(result.n)*result.k;
  require(fields[4]==elements/2&&fields[5]==elements/64*2&&fields[6]==fields[5]&&
      fields[7]==elements*4&&fields[8]==elements*2,"invalid fixture section lengths");
  result.packed.resize(fields[4]);result.scales.resize(fields[5]/2);result.biases.resize(fields[6]/2);
  result.f32Coefficients.resize(elements);result.bf16Coefficients.resize(elements);
  read(result.packed.data(),result.packed.size());read(result.scales.data(),fields[5]);
  read(result.biases.data(),fields[6]);read(result.f32Coefficients.data(),fields[7]);
  read(result.bf16Coefficients.data(),fields[8]);
  require(stream.peek()==std::char_traits<char>::eof(),"trailing fixture bytes");
  for(uint32_t n=0;n<result.n;++n)for(uint32_t k=0;k<result.k;++k) {
    const auto p=result.parameter(n,k),index=uint64_t(n)*result.k+k;
    const float product=float(result.code(n,k))*number(result.scales[p]);
    const float coefficient=product+number(result.biases[p]);
    require(std::isfinite(coefficient)&&result.f32Coefficients[index]==std::bit_cast<uint32_t>(coefficient)&&
        result.bf16Coefficients[index]==bf16(coefficient),"source coefficient certificate failed");
    ++result.sourceChecks;
  }
  result.packedSHA=digest(result.packed.data(),result.packed.size());
  result.f32SHA=digest(result.f32Coefficients.data(),result.f32Coefficients.size()*4);
  result.bf16SHA=digest(result.bf16Coefficients.data(),result.bf16Coefficients.size()*2);
  return result;
}
template<typename Word> struct Guarded {
  MetalBuffer base,view;uint64_t words,allocation;Word marker;
  Guarded(MetalBackend &backend,uint64_t count,Word mark,const char *label):words(count),marker(mark) {
    const auto before=backend.memoryStats().allocatedBytes;
    base=backend.allocateBuffer((words+2*guardWords)*sizeof(Word),BufferStorage::Shared,label);
    view=backend.view(base,guardWords*sizeof(Word),words*sizeof(Word));
    allocation=backend.memoryStats().allocatedBytes-before;
    std::fill_n(static_cast<Word *>(base.contents()),words+2*guardWords,marker);
  }
  Word *data() {return static_cast<Word *>(view.contents());}
  const Word *data()const{return static_cast<const Word *>(view.contents());}
  void guards()const {
    const auto data=static_cast<const Word *>(base.contents());
    for(uint32_t i=0;i<guardWords;++i)
      require(data[i]==marker&&data[guardWords+words+i]==marker,"GPU guard overwritten");
  }
};
struct Reference {
  uint32_t row,column;
  double exactSum=0,f32Sum=0,bf16Sum=0,factorL1=0,f32L1=0,bf16L1=0,f32Loss=0,bf16Loss=0;
};
Reference referenceCell(const Fixture &f,const uint16_t *input,uint32_t row,uint32_t column) {
  Reference r;r.row=row;r.column=column;
  for(uint32_t k=0;k<f.k;++k) {
    const double a=number(input[uint64_t(row)*f.k+k]);
    const auto p=f.parameter(column,k),index=uint64_t(column)*f.k+k;
    const double exact=f.exact(column,k),w32=std::bit_cast<float>(f.f32Coefficients[index]),
        w16=number(f.bf16Coefficients[index]);
    r.exactSum+=a*exact;r.f32Sum+=a*w32;r.bf16Sum+=a*w16;
    r.factorL1+=std::abs(a)*(std::abs(double(f.code(column,k))*number(f.scales[p]))+std::abs(number(f.biases[p])));
    r.f32L1+=std::abs(a*w32);r.bf16L1+=std::abs(a*w16);
    r.f32Loss+=std::abs(a)*std::abs(w32-exact);r.bf16Loss+=std::abs(a)*std::abs(w16-exact);
  }
  return r;
}
struct Errors {
  uint64_t cells=0,intervalViolations=0,strictViolations=0,cancellationCells=0,wrongSign=0;
  uint32_t maxULP=0;
  double maxAbsFromExact=0,exactError2=0,exactNorm2=0,maxCoefficientLossBound=0;
  void add(uint16_t actual,const Reference &r,const Fixture &f,uint32_t kind) {
    require(std::isfinite(number(actual)),"unwritten or nonfinite projection output");++cells;
    const double math=kind==1?r.f32Sum:kind==2?r.bf16Sum:r.exactSum;
    const double l1=kind==1?r.f32L1:kind==2?r.bf16L1:r.factorL1;
    const double arithmetic=f32DotBound(f.k+(kind?0:4*(f.k/64)),l1);
    const double loss=kind==1?r.f32Loss:kind==2?r.bf16Loss:0;
    const auto golden=bf16Double(math);
    const double error=double(number(actual))-number(bf16Double(r.exactSum));
    maxAbsFromExact=std::max(maxAbsFromExact,std::abs(error));
    exactError2+=error*error;exactNorm2+=double(number(bf16Double(r.exactSum)))*number(bf16Double(r.exactSum));
    maxCoefficientLossBound=std::max(maxCoefficientLossBound,loss);
    strictViolations+=!cellRelation(actual,golden,math,arithmetic).pass;
    maxULP=std::max(maxULP,bf16ULP(actual,golden));
    const auto low=orderedBF16(bf16Double(r.exactSum-arithmetic-loss));
    const auto high=orderedBF16(bf16Double(r.exactSum+arithmetic+loss));
    const auto order=orderedBF16(actual);intervalViolations+=order<low||order>high;
    cancellationCells+=std::abs(r.exactSum)<=arithmetic+loss;
    wrongSign+=r.exactSum!=0&&number(actual)!=0&&std::signbit(r.exactSum)!=std::signbit(number(actual));
  }
  double relativeL2()const{return std::sqrt(exactError2/std::max(1e-30,exactNorm2));}
  void write(std::ostream &out)const {
    out<<"{\"sampled_cells\":"<<cells<<",\"source_math_interval_violations\":"<<intervalViolations
      <<",\"strict_matrix_cell_violations\":"<<strictViolations<<",\"max_ulp_vs_own_coefficient_math\":"<<maxULP
      <<",\"max_abs_vs_source_affine_bf16\":"<<maxAbsFromExact<<",\"relative_l2_vs_source_affine_bf16\":"<<relativeL2()
      <<",\"max_input_weighted_coefficient_loss_bound\":"<<maxCoefficientLossBound
      <<",\"cancellation_dominated_cells\":"<<cancellationCells<<",\"wrong_sign_vs_source_affine_cells\":"<<wrongSign<<'}';
  }
};
double median(std::vector<double> v) {require(!v.empty(),"empty timings");std::sort(v.begin(),v.end());return v[v.size()/2];}
struct Variant {
  std::string name;uint32_t kind,m,n,sg,mode;
  Prefill4KQ27Params params{};
  std::unique_ptr<Guarded<uint16_t>> output;
  std::vector<ComputeDispatch> dispatches;
  std::vector<double> gpu,wall;
  Errors numerical;uint64_t controlDiff=0,sameGeometryTraversalDiff=0;
  bool traversalComparatorPresent=false;
  Variant(std::string value,uint32_t coefficientKind,uint32_t tileM,uint32_t tileN,
          uint32_t simdgroups,uint32_t traversal):name(std::move(value)),kind(coefficientKind),
          m(tileM),n(tileN),sg(simdgroups),mode(traversal) {}
};
DispatchSize grid(uint32_t rows,uint32_t columns,uint32_t mode) {
  if(!mode)return {columns,rows,1};if(mode==1)return {rows,columns,1};
  return {uint64_t(columns)*4,(rows+3)/4,1};
}
void selfTest() {
  uint64_t cases=0,sumLayoutCases=0,orderedPairCases=0;
  for(uint32_t r=1;r<66;++r)for(uint32_t c=1;c<34;++c)for(uint32_t mode=0;mode<3;++mode) {
    const auto g=grid(r,c,mode);std::vector<uint32_t> seen(uint64_t(r)*c);
    for(uint32_t y=0;y<g.y;++y)for(uint32_t x=0;x<g.x;++x) {
      const auto row=mode==1?x:mode==2?y*4+(x&3):y;
      const auto col=mode==1?y:mode==2?x>>2:x;
      if(row<r&&col<c)++seen[uint64_t(row)*c+col];
    }
    for(auto count:seen)require(count==1,"grid is not bijective");++cases;
  }
  for(uint32_t tile:{16u,32u,64u})for(uint32_t rows=1;rows<=65;++rows)
    for(uint32_t groups:{1u,2u,80u,96u,272u}) {
      const auto padded=extent(rows,tile);std::vector<uint32_t> seen(uint64_t(padded)*groups);
      for(uint32_t row=0;row<padded;++row)for(uint32_t group=0;group<groups;++group) {
        const auto address=uint64_t(row/tile)*tile*groups+group*tile+row%tile;
        require(address<seen.size(),"sum-layout address out of range");++seen[address];
      }
      for(auto count:seen)require(count==1,"sum layout does not cover every row/group once");
      ++sumLayoutCases;
    }
  for(uint32_t groups=1;groups<=273;++groups) {
    std::vector<uint32_t> epilogueOrder;
    for(uint32_t group=0;group<groups;group+=2) {
      epilogueOrder.push_back(group);
      if(group+1<groups)epilogueOrder.push_back(group+1);
    }
    require(epilogueOrder.size()==groups,"paired group scheduling lost a tail");
    for(uint32_t group=0;group<groups;++group)
      require(epilogueOrder[group]==group,"paired epilogues changed source group order");
    ++orderedPairCases;
  }
  require(bf16Double(1.00390625)==0x3f80,"BF16 tie rounding changed");
  const double tiny=std::ldexp(1.0,-14),absoluteProducts=2*1984*65536.0+tiny;
  const auto broad=f32DotBound(6144,absoluteProducts);
  require(broad>tiny&&!cellRelation(0,bf16Double(tiny),tiny,broad).pass,
      "near-zero strict classifier must reject lost residual");
  std::cout<<"{\"pass\":true,\"grid_cases\":"<<cases<<",\"sum_layout_cases\":"<<sumLayoutCases
    <<",\"ordered_pair_cases\":"<<orderedPairCases<<",\"near_zero_classifier_pass\":true,\"gpu_work\":false}\n";
}
} // namespace

int main(int argc,char **argv) {
  @autoreleasepool {try {
    if(argc==2&&std::string(argv[1])=="--cpu-self-test") {selfTest();return 0;}
    if(argc==3&&std::string(argv[1])=="--source-self-test") {
      const auto f=loadFixture(argv[2]);std::cout<<"{\"pass\":true,\"gpu_work\":false,\"coefficient_checks\":"
        <<f.sourceChecks<<",\"n\":"<<f.n<<",\"k\":"<<f.k<<",\"f32_sha256\":"<<splash::json::quote(f.f32SHA)<<'}'<<'\n';return 0;
    }
    require(argc==4,"usage: prefill4k-q27-oracle METALLIB SOURCE_FIXTURE REPORT | --cpu-self-test | --source-self-test FIXTURE");
    const auto f=loadFixture(argv[2]);
    const uint32_t rows=envPositive("Q27_ROWS",2048,8192),samples=envPositive("Q27_SAMPLES",7,30),
      batch=envPositive("Q27_BATCH",2,16),storageRows=extent(rows,128);
    const char *patternRaw=std::getenv("Q27_PATTERN");const std::string pattern=patternRaw?patternRaw:"random";
    require(pattern=="random"||pattern=="alternating"||pattern=="sparse"||pattern=="mixed","invalid Q27_PATTERN");
    MetalBackend backend(argv[1]);
    Guarded<uint16_t> input(backend,uint64_t(storageRows)*f.k,marker16,"Q27 padded source input");
    Guarded<uint8_t> packed(backend,f.packed.size(),0xa5,"Q27 original Q4 packed weights");
    Guarded<uint16_t> scales(backend,f.scales.size(),marker16,"Q27 source BF16 scales"),
      biases(backend,f.biases.size(),marker16,"Q27 source BF16 biases"),
      weights16(backend,f.bf16Coefficients.size(),marker16,"Q27 BF16 alternate coefficients");
    Guarded<uint32_t> weights32(backend,f.f32Coefficients.size(),marker32,"Q27 source F32 coefficients");
    Guarded<uint32_t> sums16(backend,uint64_t(storageRows)*(f.k/64),marker32,"Q27 candidate16 row sums"),
      sums32(backend,uint64_t(storageRows)*(f.k/64),marker32,"Q27 original32 row sums"),
      sums64(backend,uint64_t(storageRows)*(f.k/64),marker32,"Q27 candidate64 row sums");
    auto diagnostics=backend.allocateBuffer(4,BufferStorage::Shared,"Q27 sticky diagnostics");
    std::fill_n(input.data(),input.words,uint16_t(0));
    for(uint64_t i=0;i<uint64_t(rows)*f.k;++i) {
      const auto word=randomWord(i);
      const float value=pattern=="alternating"?(i&1?-.75f:.75f):pattern=="sparse"?
        (i%128==0?.5f:0.0f):pattern=="mixed"?
        std::ldexp((word&1?-.75f:.75f),int((word>>32)%33)-16):
        float(int32_t(word%2047)-1023)/1024.0f;
      input.data()[i]=bf16(value);
    }
    std::memcpy(packed.data(),f.packed.data(),f.packed.size());
    std::memcpy(scales.data(),f.scales.data(),f.scales.size()*2);std::memcpy(biases.data(),f.biases.data(),f.biases.size()*2);
    std::memcpy(weights16.data(),f.bf16Coefficients.data(),f.bf16Coefficients.size()*2);
    std::memcpy(weights32.data(),f.f32Coefficients.data(),f.f32Coefficients.size()*4);
    const auto inputHash=digest(input.data(),input.words*2);
    Q4PrefillParams q4{f.n,f.k};
    std::vector<Variant> variants;
    variants.push_back({"q4_control_m32n128_s4",0,32,128,4,1});
    variants.push_back({"q4_candidate_m64n128_s4",0,64,128,4,1});
    variants.push_back({"q4_original_m32n128_s8_staged",0,32,128,8,1});
    variants.push_back({"q4_candidate_m32n128_s8_direct",0,32,128,8,1});
    variants.push_back({"q4_candidate_m32n64_s2_direct",0,32,64,2,1});
    variants.push_back({"q4_candidate_m32n64_s4_direct",0,32,64,4,1});
    variants.push_back({"q4_candidate_m16n64_s2_direct",0,16,64,2,1});
    variants.push_back({"q4_candidate_m16n64_s4_direct",0,16,64,4,1});
    variants.push_back({"q4_candidate_m32n64_s4_pair",0,32,64,4,1});
    variants.push_back({"q4_candidate_m16n64_s2_pair",0,16,64,2,1});
    for(uint32_t kind=1;kind<=2;++kind)for(auto tile:std::array<std::array<uint32_t,3>,3>{{{32,128,4},{64,128,8},{128,64,8}}})
      for(uint32_t mode=0;mode<3;++mode)variants.push_back({std::string(kind==1?"f32":"bf16")+
        "_m"+std::to_string(tile[0])+"n"+std::to_string(tile[1])+"_s"+std::to_string(tile[2])+"_t"+std::to_string(mode),
        kind,tile[0],tile[1],tile[2],mode});
    const char *filter=std::getenv("Q27_FILTER");
    if(filter) {
      std::vector<std::array<uint32_t,3>> cachedComparators;
      for(const auto &v:variants)if(v.kind&&v.name.find(filter)!=std::string::npos)
        cachedComparators.push_back({v.kind,v.m,v.n});
      std::erase_if(variants,[&](const Variant &v) {
        if(v.name=="q4_control_m32n128_s4"||v.name.find(filter)!=std::string::npos)return false;
        return v.mode!=0||std::find(cachedComparators.begin(),cachedComparators.end(),
            std::array<uint32_t,3>{v.kind,v.m,v.n})==cachedComparators.end();
      });
    }
    require(variants.size()>1,"Q27_FILTER must match a candidate");
    for(auto &v:variants) {
      v.params={rows,f.k,f.n,v.m,v.n,v.mode,0,0};
      v.output=std::make_unique<Guarded<uint16_t>>(backend,uint64_t(storageRows)*f.n,marker16,v.name.c_str());
      for(uint32_t repeat=0;repeat<batch;++repeat) {
        if(v.kind==0) {
          const auto &sums=v.m==16?sums16:v.m==32?sums32:sums64;
          ComputeDispatch sum;sum.pipelineName=v.m==16?"prefill4k_q27_q4_sums16":
            v.m==32?"prefill_linear_q4_sums32":"prefill4k_q27_q4_sums64";
          sum.buffers={{0,input.view},{1,sums.view}};sum.bytes={{2,&q4,sizeof(q4)}};
          sum.threadgroups={(rows+v.m-1)/v.m,1,1};sum.threadsPerThreadgroup={256,1,1};v.dispatches.push_back(std::move(sum));
          ComputeDispatch projection;
          projection.pipelineName=v.n==64?"prefill4k_q27_q4_m"+std::to_string(v.m)+
            "n64_s"+std::to_string(v.sg)+(v.name.find("pair")!=std::string::npos?"_pair":"_direct"):
            v.m!=32?"prefill4k_q27_q4_m64n128_s4":v.sg==4?
            "prefill_linear_q4_n128_sg4":v.name.find("staged")!=std::string::npos?
            "prefill_linear_q4_n128":"prefill4k_q27_q4_m32n128_s8_direct";
          projection.buffers={{0,input.view},{1,packed.view},{2,scales.view},{3,biases.view},{4,v.output->view},{5,sums.view}};
          projection.bytes={{6,&q4,sizeof(q4)}};projection.threadgroups={(rows+v.m-1)/v.m,f.n/v.n,1};
          projection.threadsPerThreadgroup={v.sg*32,1,1};v.dispatches.push_back(std::move(projection));
        }else {
          ComputeDispatch d;d.pipelineName="prefill4k_q27_"+std::string(v.kind==1?"f32":"bf16")+
            "_m"+std::to_string(v.m)+"n"+std::to_string(v.n)+"_s"+std::to_string(v.sg);
          d.buffers={{0,input.view},{1,v.kind==1?weights32.view:weights16.view},{2,v.output->view},{3,diagnostics}};
          d.bytes={{4,&v.params,sizeof(v.params)}};d.threadgroups=grid((rows+v.m-1)/v.m,f.n/v.n,v.mode);
          d.threadsPerThreadgroup={v.sg*32,1,1};v.dispatches.push_back(std::move(d));
        }
      }
    }
    auto *status=static_cast<uint32_t *>(diagnostics.contents());
    for(uint32_t warm=0;warm<2;++warm)for(auto &v:variants) {*status=sticky;(void)backend.submitCommand(v.dispatches);require(*status==sticky,"warmup diagnostics changed");}
    for(uint32_t sample=0;sample<samples;++sample)for(uint32_t offset=0;offset<variants.size();++offset) {
      auto &v=variants[(sample+offset)%variants.size()];*status=sticky;
      const auto timing=backend.submitCommand(v.dispatches);
      require(*status==sticky&&timing.gpuSeconds>0&&timing.wallSeconds>0,"timing or diagnostics invalid");
      v.gpu.push_back(timing.gpuSeconds*1000/batch);v.wall.push_back(timing.wallSeconds*1000/batch);
    }
    input.guards();packed.guards();scales.guards();biases.guards();weights16.guards();weights32.guards();sums16.guards();sums32.guards();sums64.guards();
    require(inputHash==digest(input.data(),input.words*2)&&f.packedSHA==digest(packed.data(),f.packed.size())&&
      f.f32SHA==digest(weights32.data(),f.f32Coefficients.size()*4)&&f.bf16SHA==digest(weights16.data(),f.bf16Coefficients.size()*2)&&
      std::memcmp(scales.data(),f.scales.data(),f.scales.size()*2)==0&&std::memcmp(biases.data(),f.biases.data(),f.biases.size()*2)==0,
      "GPU mutated source operands");
    std::vector<Reference> references;
    const uint32_t nr=std::min(rows,rows<=64?rows:9u),nc=std::min(f.n,19u);
    for(uint32_t i=0;i<nr;++i)for(uint32_t j=0;j<nc;++j)
      references.push_back(referenceCell(f,input.data(),nr==1?0:uint64_t(i)*(rows-1)/(nr-1),nc==1?0:uint64_t(j)*(f.n-1)/(nc-1)));
    std::ostringstream out;out<<std::setprecision(12)<<"{\"pass\":true,\"pass_scope\":\"source coefficients, guards, finite outputs, source arithmetic interval and same-geometry traversal parity; strict matrix and native-Q4 parity reported separately\",\"model_loaded\":false,\"production_modified\":false,\"source_fixture\":"
      <<splash::json::quote(argv[2])<<",\"source_coefficient_checks\":"<<f.sourceChecks<<",\"packed_sha256\":"<<splash::json::quote(f.packedSHA)
      <<",\"f32_coefficient_sha256\":"<<splash::json::quote(f.f32SHA)<<",\"bf16_coefficient_sha256\":"<<splash::json::quote(f.bf16SHA)
      <<",\"rows\":"<<rows<<",\"k\":"<<f.k<<",\"n\":"<<f.n<<",\"input_pattern\":"<<splash::json::quote(pattern)
      <<",\"samples\":"<<samples<<",\"command_batch\":"<<batch
      <<",\"timing_scope\":\"complete projection; Q4 includes input sums; one-time CPU coefficient reconstruction and source upload excluded; native backend; no die affinity\""
      <<",\"numerical_scope\":\"independent double source affine and per-cache dots; source arithmetic interval includes measured input-weighted coefficient rounding loss; strict own-coefficient matrix gate separate; no model generation claim\""
      <<",\"bf16_cache_operand_bytes\":"<<f.bf16Coefficients.size()*2<<",\"f32_cache_operand_bytes\":"<<f.f32Coefficients.size()*4
      <<",\"actual_native_allocation_bytes\":"<<backend.memoryStats().allocatedBytes<<",\"records\":[";
    bool first=true;const uint64_t logical=uint64_t(rows)*f.n;const auto *control=variants[0].output->data();
    for(auto &v:variants) {
      v.output->guards();for(uint64_t i=0;i<logical;++i) {require(std::isfinite(number(v.output->data()[i])),"output nonfinite");v.controlDiff+=v.output->data()[i]!=control[i];}
      if(v.kind) {
        for(uint64_t i=logical;i<v.output->words;++i)
          require(v.output->data()[i]==marker16,"cached projection wrote padded output rows");
        for(const auto &other:variants)if(other.kind==v.kind&&other.m==v.m&&other.n==v.n&&!other.mode) {
          v.traversalComparatorPresent=true;
          for(uint64_t i=0;i<logical;++i)v.sameGeometryTraversalDiff+=v.output->data()[i]!=other.output->data()[i];
        }
        require(v.traversalComparatorPresent,"cached traversal comparator missing");
      }
      for(const auto &r:references)v.numerical.add(v.output->data()[uint64_t(r.row)*f.n+r.column],r,f,v.kind);
      require(v.numerical.intervalViolations==0,"source math interval failed");
      require(v.sameGeometryTraversalDiff==0,"traversal changed same-geometry coefficients or arithmetic");
      if(!first)out<<',';first=false;
      const auto gpu=median(v.gpu);out<<"{\"variant\":"<<splash::json::quote(v.name)<<",\"coefficient_kind\":"
        <<splash::json::quote(v.kind==1?"source-exact F32; alternate grouping":v.kind==2?"once-rounded BF16; numerical alternate":"original Q4 group-affine")
        <<",\"median_gpu_ms\":"<<gpu<<",\"median_wall_ms\":"<<median(v.wall)<<",\"gpu_speedup_vs_native_q4\":"<<median(variants[0].gpu)/gpu
        <<",\"full_bf16_differences_vs_native_q4\":"<<v.controlDiff<<",\"same_geometry_traversal_differences\":"<<v.sameGeometryTraversalDiff
        <<",\"same_geometry_traversal_comparator_present\":"<<(v.traversalComparatorPresent?"true":"false")
        <<",\"native_q4_bf16_exact\":"<<(v.controlDiff?"false":"true")<<",\"strict_own_coefficient_matrix_gate_pass\":"<<(v.numerical.strictViolations?"false":"true")
        <<",\"guards_and_immutable_operands_pass\":true,\"numerical\":";v.numerical.write(out);out<<'}';
      std::cerr<<v.name<<" gpu_ms="<<gpu<<" speedup="<<median(variants[0].gpu)/gpu<<" native_diff="<<v.controlDiff<<'\n';
    }
    out<<"]}\n";std::ofstream report(argv[3]);require(bool(report),"cannot open report");report<<out.str();report.close();require(bool(report),"cannot write report");
    std::cout<<"{\"pass\":true,\"variant_count\":"<<variants.size()<<",\"model_loaded\":false,\"report\":"<<splash::json::quote(argv[3])<<"}\n";return 0;
  }catch(const std::exception &error){std::cerr<<"prefill4k-q27-oracle: "<<error.what()<<'\n';return 1;}}
}
