#pragma once
#include "flash/FlashWeights.hpp"
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace splash::flash::prefill4k_q4coded {
inline double bfNumber(uint16_t word) { return double(std::bit_cast<float>(uint32_t(word) <<16)); }
inline uint16_t bfWord(float value) { const auto w =std::bit_cast<uint32_t>(value);return uint16_t((w +0x7fff +((w >>16) &1)) >>16); }
inline double gamma(uint32_t n,double unit =0x1p-24) { return double(n) *unit /(1 -double(n) *unit); }
inline double upward(double value) { return std::nextafter(value,std::numeric_limits<double>::infinity()); }
inline double addUp(double a,double b) { return upward(a +b); }
inline double mulUp(double a,double b) { return upward(a *b); }
inline double gammaUp(uint32_t n,double unit) {
  const double product=mulUp(double(n),unit);
  const double denominator=std::nextafter(1 -product,-std::numeric_limits<double>::infinity());
  return upward(product /denominator);
}
inline double underflowUp(uint32_t n,double unit,double eta) {
  const double denominator=std::nextafter(1 -mulUp(double(n),unit),-std::numeric_limits<double>::infinity());
  return upward(mulUp(double(n),eta) /denominator);
}
struct Certificate { double affine =0,sourceBF16 =0,coefficientEnvelope =0,candidateEnvelope =0,diagnosticEnvelope =0,sourceNorm =0; };
// Independent byte extraction, promoted source parameters, FP64 dots/norms.
// The envelope is absolute and stays meaningful at a zero/cancelling dot.
inline Certificate certificate(const FlashAffineProjection &p,uint32_t expert,uint32_t n,const uint16_t *input) {
  if (p.bits !=4 || p.groupSize !=64 || p.inputSize %64 || expert >=512 || n >=p.outputSize)
    throw std::invalid_argument("Q4coded certificate source geometry differs");
  const auto *w =static_cast<const uint8_t *>(p.weights->buffer.contents()) +uint64_t(expert) *p.weightExpertStrideBytes +uint64_t(n) *p.weightRowStrideBytes;
  const auto *sc =reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(p.scales->buffer.contents()) +uint64_t(expert) *p.parameterExpertStrideBytes +uint64_t(n) *p.parameterRowStrideBytes);
  const auto *bi =reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(p.biases->buffer.contents()) +uint64_t(expert) *p.parameterExpertStrideBytes +uint64_t(n) *p.parameterRowStrideBytes);
  constexpr double u =0x1p-24,eta =0x1p-150,u64=0x1p-53,eta64=std::numeric_limits<double>::denorm_min();
  Certificate result;double groupErrors =0,groupMagnitude =0,sourceMagnitude =0,referenceErrors=0,sourceNorm=0;
  for (uint32_t group =0; group <p.inputSize /64; ++group) {
    const double scale =bfNumber(sc[group]),bias =bfNumber(bi[group]);double dot =0,sum =0,Lq =0,Lx =0;
    if (!std::isfinite(scale) || !std::isfinite(bias)) throw std::runtime_error("Q4coded nonfinite source parameter");
    for (uint32_t j =0; j <64; ++j) {
      const uint32_t k =group *64 +j,q =(w[k /2] >>((k &1) *4)) &15;
      const double x =bfNumber(input[k]);if (!std::isfinite(x)) throw std::runtime_error("Q4coded nonfinite BF16 input");
      const double coefficient =double(q) *scale +bias;
      const float product =float(q) *float(scale);const float reconstructed =product +float(bias);
      if (!std::isfinite(reconstructed)) throw std::runtime_error("Q4coded source F32 reconstruction overflow");
      const double rounded =bfNumber(bfWord(reconstructed));
      if (!std::isfinite(rounded)) throw std::runtime_error("Q4coded overflowed source BF16 coefficient");
      dot +=x *q;sum +=x;Lq=addUp(Lq,std::abs(x *q));Lx=addUp(Lx,std::abs(x));
      result.sourceBF16 +=x *rounded;
      // q*s is an exact dyadic in FP64. Include RN64 coefficient addition and
      // subtraction errors before outward accumulation of the tight residual.
      const double coefficientRound=addUp(mulUp(u64,addUp(std::abs(q *scale),std::abs(bias))),eta64);
      const double residualRound=addUp(mulUp(u64,addUp(std::abs(rounded),std::abs(coefficient))),eta64);
      const double residual=addUp(std::abs(rounded -coefficient),addUp(coefficientRound,residualRound));
      result.coefficientEnvelope=addUp(result.coefficientEnvelope,mulUp(std::abs(x),residual));
      sourceMagnitude=addUp(sourceMagnitude,mulUp(std::abs(x),addUp(std::abs(q *scale),std::abs(bias))));
      sourceNorm=addUp(sourceNorm,std::abs(x *rounded));
    }
    result.affine +=scale *dot +bias *sum;
    const double magnitude=addUp(mulUp(std::abs(scale),Lq),mulUp(std::abs(bias),Lx));
    const auto groupEnvelope=[&](double unit,double underflow) {
      const double eD=addUp(mulUp(gammaUp(64,unit),Lq),underflowUp(64,unit,underflow));
      const double eT=addUp(mulUp(gammaUp(64,unit),Lx),underflowUp(64,unit,underflow));
      const double ep=addUp(addUp(mulUp(mulUp(addUp(1,unit),std::abs(scale)),eD),mulUp(mulUp(unit,std::abs(scale)),Lq)),underflow);
      const double ev=addUp(addUp(mulUp(mulUp(addUp(1,unit),std::abs(bias)),eT),mulUp(mulUp(unit,std::abs(bias)),Lx)),underflow);
      return addUp(addUp(mulUp(addUp(1,unit),addUp(ep,ev)),mulUp(unit,magnitude)),underflow);
    };
    groupErrors=addUp(groupErrors,groupEnvelope(u,eta));referenceErrors=addUp(referenceErrors,groupEnvelope(u64,eta64));
    groupMagnitude=addUp(groupMagnitude,magnitude);
  }
  const uint32_t groups =p.inputSize /64;
  result.candidateEnvelope=addUp(addUp(mulUp(addUp(1,gammaUp(groups,u)),groupErrors),mulUp(gammaUp(groups,u),groupMagnitude)),underflowUp(groups,u,eta));
  const double referenceEnvelope=addUp(addUp(mulUp(addUp(1,gammaUp(groups,u64)),referenceErrors),mulUp(gammaUp(groups,u64),groupMagnitude)),underflowUp(groups,u64,eta64));
  const double sourceReferenceEnvelope=addUp(mulUp(gammaUp(p.inputSize,u64),sourceNorm),underflowUp(p.inputSize,u64,eta64));
  // Any finite got with true candidate error<=E has magnitude at most
  // sourceMagnitude+E. Include final FP64 diagnostic subtraction as well.
  const double subtraction=addUp(mulUp(u64,addUp(addUp(sourceMagnitude,result.candidateEnvelope),addUp(sourceMagnitude,referenceEnvelope))),eta64);
  result.diagnosticEnvelope=addUp(referenceEnvelope,subtraction);
  result.candidateEnvelope=addUp(result.candidateEnvelope,result.diagnosticEnvelope);
  const double sourceSubtraction=addUp(mulUp(u64,addUp(addUp(sourceMagnitude,result.candidateEnvelope),addUp(sourceNorm,sourceReferenceEnvelope))),eta64);
  result.coefficientEnvelope=addUp(result.coefficientEnvelope,addUp(sourceReferenceEnvelope,sourceSubtraction));
  result.sourceNorm=sourceNorm;
  return result;
}
} // namespace splash::flash::prefill4k_q4coded
