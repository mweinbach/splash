#pragma once
#include "metal/MetalBackend.hpp"
#include <array>
#include <cstdint>
#include <stdexcept>
namespace splash::flash::r1_capture {
struct Layer {
  metal::MetalBuffer hidden,ids,activated,down;
};
struct Context {
  std::array<Layer,48> layers;
  std::array<bool,48> seen{};
};
inline thread_local Context *current=nullptr;
inline bool active(){return current!=nullptr;}
inline Layer &get(uint32_t layer){
  if(!current||layer>=48)throw std::logic_error("R1 diagnostic capture scope/geometry invalid");
  current->seen[layer]=true;return current->layers[layer];
}
class Scoped final {
  Context *previous;
public:
  explicit Scoped(Context &context):previous(current){
    if(current)throw std::logic_error("nested R1 capture prohibited");
    context.seen.fill(false);current=&context;
  }
  ~Scoped(){current=previous;}
};
} // namespace splash::flash::r1_capture
