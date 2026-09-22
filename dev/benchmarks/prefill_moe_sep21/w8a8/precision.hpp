#pragma once
// Independent CPU certificates for a NUMERICAL activation-quantized producer.
// Original BF16 A and original signed-I8 B are the FP64 mathematical reference.
// No result here establishes model quality or equivalence to original A math.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace splash::flash::prefill_moe_w8a8::precision {
inline constexpr double kMaximumProducerRelativeL2=0.05;
inline constexpr double kMinimumProducerCosine=0.9985;
inline constexpr uint32_t kMaximumK=2560;
inline constexpr uint64_t kMaximumIntegerMagnitude=127ULL*128*kMaximumK;
inline float number(uint16_t bits) { return std::bit_cast<float>(uint32_t(bits)<<16); }
inline uint16_t bf16(float value) {
  const auto bits=std::bit_cast<uint32_t>(value);
  return uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
}
inline bool sourceFinite(uint16_t value) { return (value&0x7f80u)!=0x7f80u; }
inline bool finitePositive(float value) { return std::isfinite(value) &&value>0; }
inline bool powerOfTwo(float value) {
  const uint32_t bits=std::bit_cast<uint32_t>(value)&0x7fffffffu;
  if (!bits ||bits>=0x7f800000u) return false;
  return bits>=0x00800000u ? (bits&0x007fffffu)==0 :std::has_single_bit(bits);
}
inline double floatUlp(float value) {
  if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
  const float magnitude=std::abs(value);
  if (!magnitude) return std::numeric_limits<float>::denorm_min();
  const float next=std::nextafter(magnitude,std::numeric_limits<float>::infinity());
  return std::isfinite(next) ? double(next)-double(magnitude) :
      double(magnitude)-double(std::nextafter(magnitude,0.0f));
}
inline int roundEven(double value) {
  if (value>=127) return 127;
  if (value<=-127) return -127;
  if (!std::isfinite(value)) return 0;
  const double magnitude=std::abs(value),base=std::floor(magnitude),fraction=magnitude-base;
  const int whole=int(base);
  const int rounded=whole+int(fraction>0.5 ||(fraction==0.5 &&(whole&1)));
  return std::signbit(value) ? -rounded :rounded;
}
inline float roundedMultiply(float a,float b) {
  volatile float rounded=a*b;
  return rounded;
}
inline void writeNumber(std::ostream &out,double value) {
  if (std::isfinite(value)) out<<std::setprecision(17)<<value;else out<<"null";
}

struct QuantizationReport final {
  uint64_t rows=0,values=0,sourceNonfinite=0,zeroRows=0,codeOutsideRange=0;
  uint64_t invalidScales=0,scaleFaults=0,scaleBitMismatches=0,rneFaults=0,rneBoundaryAmbiguities=0;
  double maxAbsError=0,maxHalfStepViolation=0,maxScaleAbsoluteError=0;
  bool pass() const { return !codeOutsideRange &&!invalidScales &&!scaleFaults &&!rneFaults; }
  void write(std::ostream &out) const {
    out<<"{\"pass\":"<<(pass() ? "true" :"false")<<",\"rows\":"<<rows<<",\"values\":"<<values
        <<",\"source_nonfinite_sanitized\":"<<sourceNonfinite<<",\"zero_rows\":"<<zeroRows
        <<",\"codes_outside_minus127_to127\":"<<codeOutsideRange<<",\"invalid_scales\":"<<invalidScales
        <<",\"scale_faults\":"<<scaleFaults<<",\"scale_bit_mismatches\":"<<scaleBitMismatches
        <<",\"rne_faults\":"<<rneFaults<<",\"rne_division_boundary_ambiguities\":"<<rneBoundaryAmbiguities
        <<",\"maximum_absolute_input_quantization_error\":";writeNumber(out,maxAbsError);
    out<<",\"maximum_half_step_violation\":";writeNumber(out,maxHalfStepViolation);
    out<<",\"maximum_scale_absolute_error\":";writeNumber(out,maxScaleAbsoluteError);
    out<<",\"reference\":\"finite BF16 values, nonfinite sanitized to0; maxabs/127 F32, zero row scale1; RNE ties even; division tolerance3 F32 ULP only near half boundaries\"}";
  }
};

inline QuantizationReport certifyQuantizedRows(const uint16_t *source,const int8_t *quantized,
    const float *rowScales,uint32_t rows,uint32_t K) {
  if (!source ||!quantized ||!rowScales ||!rows ||!K ||K>kMaximumK)
    throw std::invalid_argument("W8A8 quantized row certificate geometry invalid");
  QuantizationReport report;report.rows=rows;report.values=uint64_t(rows)*K;
  for (uint32_t row=0;row<rows;++row) {
    float maximum=0;
    for (uint32_t k=0;k<K;++k) {
      const auto value=source[uint64_t(row)*K+k];
      if (sourceFinite(value)) maximum=std::max(maximum,std::abs(number(value)));
      else ++report.sourceNonfinite;
    }
    const float expected=maximum>0 ? maximum/127.0f :1.0f,scale=rowScales[row];
    report.zeroRows+=maximum==0;
    if (!finitePositive(scale)) { ++report.invalidScales;continue; }
    const double scaleError=std::abs(double(scale)-double(expected));
    report.maxScaleAbsoluteError=std::max(report.maxScaleAbsoluteError,scaleError);
    report.scaleBitMismatches+=std::bit_cast<uint32_t>(scale)!=std::bit_cast<uint32_t>(expected);
    if (!finitePositive(expected) ||scaleError>3*floatUlp(expected)) ++report.scaleFaults;
    for (uint32_t k=0;k<K;++k) {
      const auto at=uint64_t(row)*K+k;
      const bool finite=sourceFinite(source[at]);
      const float value=finite ? number(source[at]) :0.0f;
      const int actual=int(quantized[at]);
      if (actual<-127 ||actual>127) ++report.codeOutsideRange;
      if (!finite) { if (actual) ++report.rneFaults;continue; }
      const float ratio=value/scale;
      const int expectedCode=roundEven(double(ratio));
      if (actual!=expectedCode) {
        // MSL float division is not an exact rational operation. At half-step
        // boundaries allow only codes produced by a3-ULP ratio neighborhood.
        // Exact power-of-two division, including scale1 tie fixtures, is strict.
        const double radius=powerOfTwo(scale) ? 0 :3*floatUlp(ratio);
        const int low=roundEven(double(ratio)-radius),high=roundEven(double(ratio)+radius);
        if (radius>0 &&actual>=std::min(low,high) &&actual<=std::max(low,high))
          ++report.rneBoundaryAmbiguities;
        else ++report.rneFaults;
      }
      const double error=std::abs(double(value)-double(actual)*double(scale));
      report.maxAbsError=std::max(report.maxAbsError,error);
      const double violation=error-double(scale)*0.5;
      report.maxHalfStepViolation=std::max(report.maxHalfStepViolation,violation);
      const double allowance=4*floatUlp(value)+4*floatUlp(scale)*127;
      if (violation>allowance) ++report.rneFaults;
    }
  }
  return report;
}

struct ProjectionReport final {
  uint32_t K=0;
  uint64_t sourceNonfinite=0,codesOutsideRange=0;
  int64_t expectedIntegerDot=0,actualIntegerDot=0;
  bool scalesValid=false,integerExact=false,scaledBitsExact=false,scaledWithinEnvelope=false,withinOriginalEnvelope=false;
  double originalScaledF64=0,approximateScaledF64=0,quantizationEnvelope=0;
  double floatingEnvelope=0,referenceEnvelope=0,rawScaledAbsoluteErrorFromOriginal=0;
  double rawScaledAbsoluteErrorFromApproximate=0,expectedScaledF32=0,actualScaledF32=0;
  bool pass() const { return !sourceNonfinite &&!codesOutsideRange &&scalesValid &&integerExact &&scaledWithinEnvelope &&withinOriginalEnvelope; }
  void write(std::ostream &out) const {
    out<<"{\"pass\":"<<(pass() ? "true" :"false")<<",\"K\":"<<K
        <<",\"expected_exact_i64_integer_dot\":"<<expectedIntegerDot<<",\"actual_gpu_i32_integer_dot\":"<<actualIntegerDot
        <<",\"integer_dot_exact\":"<<(integerExact ? "true" :"false")
        <<",\"scaled_f32_bits_exact\":"<<(scaledBitsExact ? "true" :"false")
        <<",\"scaled_f32_within_rounding_envelope\":"<<(scaledWithinEnvelope ? "true" :"false")
        <<",\"within_original_fp64_quantization_envelope\":"<<(withinOriginalEnvelope ? "true" :"false")
        <<",\"source_nonfinite\":"<<sourceNonfinite<<",\"codes_outside_minus127_to127\":"<<codesOutsideRange
        <<",\"scales_valid\":"<<(scalesValid ? "true" :"false")<<",\"original_scaled_fp64\":";writeNumber(out,originalScaledF64);
    out<<",\"quantized_scaled_fp64\":";writeNumber(out,approximateScaledF64);
    out<<",\"absolute_quantization_envelope\":";writeNumber(out,quantizationEnvelope);
    out<<",\"absolute_floating_point_envelope\":";writeNumber(out,floatingEnvelope);
    out<<",\"fp64_reference_envelope\":";writeNumber(out,referenceEnvelope);
    out<<",\"raw_scaled_abs_error_from_original\":";writeNumber(out,rawScaledAbsoluteErrorFromOriginal);
    out<<",\"raw_scaled_abs_error_from_quantized_fp64\":";writeNumber(out,rawScaledAbsoluteErrorFromApproximate);
    out<<",\"expected_staged_scaled_f32\":";writeNumber(out,expectedScaledF32);
    out<<",\"actual_gpu_scaled_f32\":";writeNumber(out,actualScaledF32);
    out<<",\"reference\":\"sum original BF16 A times original signed-I8 B in FP64, original F32 weight scale; independent exact I64 quantized dot and staged F32 dequantization; no original-math or model-quality equivalence claim\"}";
  }
};

inline ProjectionReport certifyProjection(const uint16_t *originalA,const int8_t *quantizedA,
    float activationScale,const int8_t *originalB,uint32_t K,float weightScale,
    int32_t actualIntegerDot,float actualScaledF32) {
  if (!originalA ||!quantizedA ||!originalB ||!K ||K>kMaximumK)
    throw std::invalid_argument("W8A8 projection certificate geometry invalid");
  ProjectionReport report;report.K=K;report.actualIntegerDot=actualIntegerDot;report.actualScaledF32=actualScaledF32;
  report.scalesValid=finitePositive(activationScale) &&finitePositive(weightScale);
  double originalDot=0,compensation=0,sumAbsOriginal=0,quantizationAbs=0;
  for (uint32_t k=0;k<K;++k) {
    const double x=number(originalA[k]);const int q=int(quantizedA[k]),w=int(originalB[k]);
    report.codesOutsideRange+=q<-127 ||q>127;
    if (!std::isfinite(x)) { ++report.sourceNonfinite;continue; }
    const double product=x*double(w),corrected=product-compensation,next=originalDot+corrected;
    compensation=(next-originalDot)-corrected;originalDot=next;
    sumAbsOriginal+=std::abs(product);
    report.expectedIntegerDot+=int64_t(q)*int64_t(w);
    quantizationAbs+=std::abs(x-double(q)*double(activationScale))*std::abs(double(w));
  }
  const uint64_t safeBound=uint64_t(K)*127*128;
  report.integerExact=std::abs(report.expectedIntegerDot)<=int64_t(safeBound) &&
      report.expectedIntegerDot>=std::numeric_limits<int32_t>::min() &&
      report.expectedIntegerDot<=std::numeric_limits<int32_t>::max() &&
      report.actualIntegerDot==report.expectedIntegerDot;
  report.originalScaledF64=originalDot*double(weightScale);
  report.approximateScaledF64=double(report.expectedIntegerDot)*double(activationScale)*double(weightScale);
  report.quantizationEnvelope=quantizationAbs*std::abs(double(weightScale));
  const float integerAsFloat=float(report.expectedIntegerDot);
  const float middle=roundedMultiply(integerAsFloat,activationScale);
  const float scaled=roundedMultiply(middle,weightScale);
  report.expectedScaledF32=scaled;
  report.scaledBitsExact=std::bit_cast<uint32_t>(scaled)==std::bit_cast<uint32_t>(actualScaledF32);
  const double conversionError=std::abs(double(integerAsFloat)-double(report.expectedIntegerDot));
  const double firstMultiplyError=std::abs(double(middle)-double(integerAsFloat)*double(activationScale));
  const double secondMultiplyError=std::abs(double(scaled)-double(middle)*double(weightScale));
  const double stagedRounding=(conversionError*std::abs(double(activationScale))+firstMultiplyError)*
      std::abs(double(weightScale))+secondMultiplyError;
  //4 ULPs per F32 stage is conservative for implementation float arithmetic;
  // exact GPU I32 is checked independently and cannot hide within this bound.
  const double executionAllowance=4*floatUlp(integerAsFloat)*std::abs(double(activationScale))*
      std::abs(double(weightScale))+4*floatUlp(middle)*std::abs(double(weightScale))+4*floatUlp(scaled);
  report.floatingEnvelope=stagedRounding+executionAllowance;
  const double epsilon=std::numeric_limits<double>::epsilon(),gamma=double(K)*epsilon/(1-double(K)*epsilon);
  report.referenceEnvelope=gamma*sumAbsOriginal*std::abs(double(weightScale))+
      8*epsilon*(std::abs(report.originalScaledF64)+std::abs(report.approximateScaledF64)+report.quantizationEnvelope);
  report.rawScaledAbsoluteErrorFromOriginal=std::abs(double(actualScaledF32)-report.originalScaledF64);
  report.rawScaledAbsoluteErrorFromApproximate=std::abs(double(actualScaledF32)-report.approximateScaledF64);
  report.scaledWithinEnvelope=std::isfinite(actualScaledF32) &&std::isfinite(scaled) &&
      std::abs(double(actualScaledF32)-double(scaled))<=executionAllowance+report.referenceEnvelope;
  report.withinOriginalEnvelope=std::isfinite(actualScaledF32) &&
      report.rawScaledAbsoluteErrorFromOriginal<=report.quantizationEnvelope+report.floatingEnvelope+report.referenceEnvelope;
  return report;
}

inline void cpuSelfTest() {
  static_assert(kMaximumIntegerMagnitude==41615360 &&kMaximumIntegerMagnitude<uint64_t(std::numeric_limits<int32_t>::max()));
  const auto require=[](bool value) { if (!value) throw std::logic_error("W8A8 independent precision certificate self-test failed"); };
  for (const auto &[value,expected]:std::array<std::pair<double,int>,8>{{
       {0.5,0},{1.5,2},{2.5,2},{3.5,4},{-0.5,0},{-1.5,-2},{-2.5,-2},{-3.5,-4}}})
    require(roundEven(value)==expected);
  std::array<uint16_t,8> values{bf16(127.0f),bf16(-127.0f),bf16(0.5f),bf16(1.5f),bf16(-0.5f),bf16(-1.5f),0x7f80,0x7fc0};
  std::array<int8_t,8> codes{127,-127,0,2,0,-2,0,0};float scale=1;
  auto quant=certifyQuantizedRows(values.data(),codes.data(),&scale,1,values.size());
  require(quant.pass() &&quant.sourceNonfinite==2);
  codes[2]=1;require(!certifyQuantizedRows(values.data(),codes.data(),&scale,1,values.size()).pass());codes[2]=0;
  std::array<uint16_t,8> zeros{};std::array<int8_t,8> zeroCodes{};
  require(certifyQuantizedRows(zeros.data(),zeroCodes.data(),&scale,1,zeros.size()).pass());
  const std::array<uint16_t,4> a{bf16(0.5f),bf16(1.5f),bf16(-0.5f),bf16(-1.5f)};
  const std::array<int8_t,4> q{0,2,0,-2},w{127,-128,3,-7};
  const int32_t dot=-242;const float weightScale=0.03125f,scaled=roundedMultiply(float(dot),weightScale);
  const auto projection=certifyProjection(a.data(),q.data(),1,w.data(),a.size(),weightScale,dot,scaled);
  require(projection.pass() &&projection.integerExact &&projection.scaledBitsExact);
  require(!certifyProjection(a.data(),q.data(),1,w.data(),a.size(),weightScale,dot+1,scaled).pass());
  require(!certifyProjection(a.data(),q.data(),1,w.data(),a.size(),weightScale,dot,scaled+2).pass());
  require(!certifyProjection(a.data(),q.data(),0,w.data(),a.size(),weightScale,dot,0).pass());
  std::vector<uint16_t> maximum(kMaximumK,bf16(127.0f));
  std::vector<int8_t> maximumQ(kMaximumK,127),minimumB(kMaximumK,-128);
  const int32_t bound=-int32_t(kMaximumIntegerMagnitude);
  const float largeScaled=roundedMultiply(float(bound),weightScale);
  require(certifyProjection(maximum.data(),maximumQ.data(),1,minimumB.data(),kMaximumK,weightScale,bound,largeScaled).pass());
}
} // namespace splash::flash::prefill_moe_w8a8::precision
