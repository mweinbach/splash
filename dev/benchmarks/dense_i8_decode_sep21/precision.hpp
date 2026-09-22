#pragma once
// NUMERICAL weight requantization certificates. BF16 inputs remain unchanged.
// The mathematical reference is the original saved F32 coefficient matrix.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace splash::flash::dense_i8_decode::precision {
inline constexpr double kMaximumRelativeL2=0.02,kMinimumCosine=0.9998;
inline constexpr uint32_t kMaximumK=32768;
inline float number(uint16_t v) {return std::bit_cast<float>(uint32_t(v)<<16);}
inline uint16_t bf16(float value) {
  const auto bits=std::bit_cast<uint32_t>(value);
  return uint16_t((bits+0x7fff+((bits>>16)&1u))>>16);
}
inline bool positive(float v) {return std::isfinite(v) &&v>0;}
inline float rowScale(float maximum) {
  if (!std::isfinite(maximum) ||maximum<0) throw std::invalid_argument("nonfinite dense coefficient row maximum");
  return maximum>0 ? std::max(maximum/127.0f,std::numeric_limits<float>::min()) :1.0f;
}
inline double ulp(float value) {
  if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
  const auto magnitude=std::abs(value);
  if (!magnitude) return std::numeric_limits<float>::denorm_min();
  const auto next=std::nextafter(magnitude,std::numeric_limits<float>::infinity());
  return std::isfinite(next) ? double(next)-magnitude :double(magnitude)-std::nextafter(magnitude,0.0f);
}
inline int roundEven(double value) {
  if (value>=127) return 127;if (value<=-127) return -127;
  if (!std::isfinite(value)) return 0;
  const double magnitude=std::abs(value),base=std::floor(magnitude),fraction=magnitude-base;
  const int whole=int(base),rounded=whole+int(fraction>0.5 ||(fraction==0.5 &&(whole&1)));
  return std::signbit(value) ? -rounded :rounded;
}
inline void writeNumber(std::ostream &out,double value) {
  if (std::isfinite(value)) out<<std::setprecision(17)<<value;else out<<"null";
}
struct CoefficientReport final {
  uint64_t elements=0,rows=0,nonfinite=0,zeroRows=0,codesOutsideRange=0,scaleFaults=0,rneFaults=0;
  uint64_t rneBoundaryAmbiguities=0;
  double maximumAbsoluteError=0,maximumHalfStepViolation=0,squaredError=0,squaredReference=0;
  bool pass() const {return !nonfinite &&!codesOutsideRange &&!scaleFaults &&!rneFaults;}
  double relativeL2() const {return std::sqrt(squaredError/std::max(squaredReference,1e-300));}
  void write(std::ostream &out) const {
    out<<"{\"pass\":"<<(pass() ? "true" :"false")<<",\"elements\":"<<elements<<",\"rows\":"<<rows
       <<",\"source_nonfinite\":"<<nonfinite<<",\"zero_rows\":"<<zeroRows<<",\"codes_outside_minus127_to127\":"<<codesOutsideRange
       <<",\"scale_faults\":"<<scaleFaults<<",\"rne_faults\":"<<rneFaults<<",\"division_half_boundary_ambiguities\":"<<rneBoundaryAmbiguities
       <<",\"maximum_absolute_coefficient_error\":";writeNumber(out,maximumAbsoluteError);
    out<<",\"maximum_half_step_violation\":";writeNumber(out,maximumHalfStepViolation);
    out<<",\"coefficient_relative_l2\":";writeNumber(out,relativeL2());
    out<<",\"weight_requantization\":true,\"activation_quantization\":false,\"policy\":\"F32 per-output-row maxabs/127 with F32 minimum-normal scale floor; zeros scale1; RNE signed-I8 clamp127\"}";
  }
};
inline CoefficientReport certifyCoefficients(const float *original,const int8_t *codes,const float *scales,
    uint32_t N,uint32_t K) {
  if (!original ||!codes ||!scales ||!N ||!K ||K>kMaximumK) throw std::invalid_argument("dense I8 coefficient certificate shape invalid");
  CoefficientReport r;r.rows=N;r.elements=uint64_t(N)*K;
  for (uint32_t n=0;n<N;++n) {
    float maximum=0;
    for (uint32_t k=0;k<K;++k) {
      const auto value=original[uint64_t(n)*K+k];
      if (!std::isfinite(value)) {++r.nonfinite;continue;}
      maximum=std::max(maximum,std::abs(value));
    }
    r.zeroRows+=maximum==0;const float expected=rowScale(maximum),scale=scales[n];
    if (!positive(scale)) {++r.scaleFaults;continue;}
    if (std::abs(double(scale)-expected)>3*ulp(expected)) ++r.scaleFaults;
    for (uint32_t k=0;k<K;++k) {
      const auto at=uint64_t(n)*K+k;const float value=original[at];const int code=int(codes[at]);
      r.codesOutsideRange+=code<-127 ||code>127;
      if (!std::isfinite(value)) continue;
      const float ratio=value/scale;const int expectedCode=roundEven(ratio);
      if (code!=expectedCode) {
        const double radius=3*ulp(ratio);const int lo=roundEven(double(ratio)-radius),hi=roundEven(double(ratio)+radius);
        const auto bits=std::bit_cast<uint32_t>(scale)&0x7fffffffu;
        const bool exactPowerOfTwo=bits>=0x00800000u &&(bits&0x7fffffu)==0;
        if (!exactPowerOfTwo &&code>=std::min(lo,hi) &&code<=std::max(lo,hi)) ++r.rneBoundaryAmbiguities;
        else ++r.rneFaults;
      }
      const double error=double(code)*scale-double(value);r.squaredError+=error*error;r.squaredReference+=double(value)*value;
      r.maximumAbsoluteError=std::max(r.maximumAbsoluteError,std::abs(error));
      const double violation=std::abs(error)-double(scale)*0.5;
      r.maximumHalfStepViolation=std::max(r.maximumHalfStepViolation,violation);
      if (violation>4*ulp(value)+4*ulp(scale)*127) ++r.rneFaults;
    }
  }
  return r;
}
struct ProjectionReport final {
  uint32_t K=0;uint64_t nonfinite=0,codesOutsideRange=0;
  bool scaleValid=false,withinQuantizedEnvelope=false,withinOriginalEnvelope=false;
  double originalF64=0,quantizedF64=0,absoluteCoefficientQuantizationEnvelope=0;
  double accumulationEnvelope=0,lateScaleEnvelope=0,referenceEnvelope=0;
  double actualScaledF32=0,absoluteErrorFromOriginal=0,absoluteErrorFromQuantized=0;
  bool pass() const {return !nonfinite &&!codesOutsideRange &&scaleValid &&withinQuantizedEnvelope &&withinOriginalEnvelope;}
  void write(std::ostream &out) const {
    out<<"{\"pass\":"<<(pass() ? "true" :"false")<<",\"K\":"<<K<<",\"source_nonfinite\":"<<nonfinite
       <<",\"codes_outside_minus127_to127\":"<<codesOutsideRange<<",\"row_scale_valid\":"<<(scaleValid ? "true" :"false")
       <<",\"within_quantized_fp64_f32_rounding_envelope\":"<<(withinQuantizedEnvelope ? "true" :"false")
       <<",\"within_original_coefficient_fp64_weight_quantization_envelope\":"<<(withinOriginalEnvelope ? "true" :"false")
       <<",\"original_f32_coefficient_dot_fp64\":";writeNumber(out,originalF64);
    out<<",\"quantized_coefficient_dot_fp64\":";writeNumber(out,quantizedF64);
    out<<",\"absolute_coefficient_quantization_envelope\":";writeNumber(out,absoluteCoefficientQuantizationEnvelope);
    out<<",\"f32_accumulation_envelope\":";writeNumber(out,accumulationEnvelope);
    out<<",\"f32_late_scale_envelope\":";writeNumber(out,lateScaleEnvelope);
    out<<",\"fp64_reference_envelope\":";writeNumber(out,referenceEnvelope);
    out<<",\"actual_scaled_f32\":";writeNumber(out,actualScaledF32);
    out<<",\"absolute_error_from_original_fp64\":";writeNumber(out,absoluteErrorFromOriginal);
    out<<",\"absolute_error_from_quantized_fp64\":";writeNumber(out,absoluteErrorFromQuantized);
    out<<",\"original_math_equivalence\":false,\"model_quality_qualified\":false}";
  }
};
inline ProjectionReport certifyProjection(const uint16_t *input,const float *originalRow,const int8_t *codedRow,
    uint32_t K,float scale,float actualRawScaledF32) {
  if (!input ||!originalRow ||!codedRow ||!K ||K>kMaximumK) throw std::invalid_argument("dense I8 projection certificate shape invalid");
  ProjectionReport r;r.K=K;r.scaleValid=positive(scale);r.actualScaledF32=actualRawScaledF32;
  double original=0,originalCorrection=0,coded=0,codedCorrection=0,absoluteOriginal=0,absoluteCoded=0,quantization=0;
  for (uint32_t k=0;k<K;++k) {
    const double x=number(input[k]),w=originalRow[k];const int c=int(codedRow[k]);r.codesOutsideRange+=c<-127 ||c>127;
    if (!std::isfinite(x) ||!std::isfinite(w)) {++r.nonfinite;continue;}
    const double pOriginal=x*w,pCoded=x*double(c);
    const double a=pOriginal-originalCorrection,b=original+a;originalCorrection=(b-original)-a;original=b;
    const double c1=pCoded-codedCorrection,d=coded+c1;codedCorrection=(d-coded)-c1;coded=d;
    absoluteOriginal+=std::abs(pOriginal);absoluteCoded+=std::abs(pCoded);
    quantization+=std::abs(x)*std::abs(w-double(c)*double(scale));
  }
  r.originalF64=original;r.quantizedF64=coded*double(scale);r.absoluteCoefficientQuantizationEnvelope=quantization;
  const double u=0x1p-24,gamma=double(K)*u/(1-double(K)*u);
  // BF16 times an I8 code is exactly representable in F32 for ordinary finite
  // inputs; gamma(K) conservatively covers arbitrary F32 accumulation order.
  r.accumulationEnvelope=gamma*absoluteCoded*std::abs(double(scale));
  const float codedAsFloat=float(coded),scaledAsFloat=float(double(codedAsFloat)*double(scale));
  r.lateScaleEnvelope=4*ulp(codedAsFloat)*std::abs(double(scale))+4*ulp(scaledAsFloat);
  const double eps=std::numeric_limits<double>::epsilon(),gamma64=double(K)*eps/(1-double(K)*eps);
  r.referenceEnvelope=gamma64*(absoluteOriginal+absoluteCoded*std::abs(double(scale)))+
      8*eps*(std::abs(r.originalF64)+std::abs(r.quantizedF64)+quantization);
  r.absoluteErrorFromOriginal=std::abs(double(actualRawScaledF32)-r.originalF64);
  r.absoluteErrorFromQuantized=std::abs(double(actualRawScaledF32)-r.quantizedF64);
  const double floating=r.accumulationEnvelope+r.lateScaleEnvelope+r.referenceEnvelope;
  r.withinQuantizedEnvelope=std::isfinite(actualRawScaledF32) &&r.absoluteErrorFromQuantized<=floating;
  r.withinOriginalEnvelope=std::isfinite(actualRawScaledF32) &&r.absoluteErrorFromOriginal<=quantization+floating;
  return r;
}
inline void cpuSelfTest() {
  const auto require=[](bool ok) {if (!ok) throw std::logic_error("dense I8 weight requantization precision certificate self-test failed");};
  std::array<float,8> weights{127,-127,.5f,1.5f,-.5f,-1.5f,0,0};
  std::array<int8_t,8> codes{127,-127,0,2,0,-2,0,0};float scale=1;
  require(certifyCoefficients(weights.data(),codes.data(),&scale,1,weights.size()).pass());
  codes[2]=1;require(!certifyCoefficients(weights.data(),codes.data(),&scale,1,weights.size()).pass());codes[2]=0;
  std::array<float,8> zeros{};std::array<int8_t,8> zeroCodes{};
  require(certifyCoefficients(zeros.data(),zeroCodes.data(),&scale,1,zeros.size()).pass());
  const std::array<uint16_t,8> input{bf16(1),bf16(-.5),bf16(2),bf16(-3),bf16(-4),bf16(5),0,0};
  double dot=0;for (uint32_t k=0;k<input.size();++k) dot+=number(input[k])*codes[k];
  require(certifyProjection(input.data(),weights.data(),codes.data(),input.size(),scale,float(dot)).pass());
  require(!certifyProjection(input.data(),weights.data(),codes.data(),input.size(),scale,float(dot)+5).pass());
  require(!certifyProjection(input.data(),weights.data(),codes.data(),input.size(),0,0).pass());
  require(rowScale(0)==1 &&rowScale(std::numeric_limits<float>::denorm_min())==std::numeric_limits<float>::min());
  for (const auto &[v,e]:std::array<std::pair<double,int>,4>{{{.5,0},{1.5,2},{-.5,0},{-1.5,-2}}}) require(roundEven(v)==e);
}
} // namespace splash::flash::dense_i8_decode::precision
