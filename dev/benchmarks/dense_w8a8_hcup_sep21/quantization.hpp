#pragma once
#include "../dense_w8a8_sep21/quantization.hpp"
namespace splash::hcup_w8a8 {
namespace q = splash::dense_w8a8;
inline float f32Scale(const float *source, uint32_t k) {
  if (!source || !k) throw std::invalid_argument("F32 coefficient row is empty");
  float maximum = 0;
  for (uint32_t i = 0; i < k; ++i) {
    if (!std::isfinite(source[i])) throw std::invalid_argument("nonfinite original F32 coefficient");
    maximum = std::max(maximum,std::abs(source[i]));
  }
  const float scale = maximum == 0 ? 1 : maximum / 127.0f;
  q::validateScale(scale); return scale;
}
inline void quantizeF32(const float *source, uint32_t k, int8_t *codes,
                        float &scale, q::QuantError &error) {
  scale = f32Scale(source,k);
  for (uint32_t i = 0; i < k; ++i) {
    codes[i] = q::symmetricCode(source[i],scale);
    error.add(double(source[i]),double(codes[i])*scale,codes[i],q::symmetricClipped(source[i],scale));
  }
}
}
