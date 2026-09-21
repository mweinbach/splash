// CPU-only Metal ABI inspection. Creates functions/pipelines, never a queue,
// encoder, command buffer, or GPU submission.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) return 2;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    auto lib = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
    if (!lib) { std::cerr << error.description.UTF8String << std::endl; return 1; }
    auto fn = [lib newFunctionWithName:@"flash_ple_hash_gather128"];
    MTLComputePipelineReflection *reflection = nil;
    auto pipeline = [device newComputePipelineStateWithFunction:fn
        options:MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo
        reflection:&reflection error:&error];
    if (!pipeline) { std::cerr << error.description.UTF8String << std::endl; return 1; }
    auto encoder = [fn newArgumentEncoderWithBufferIndex:0];
    std::cout << "encodedLength=" << encoder.encodedLength << std::endl;
    for (id<MTLBinding> binding in reflection.bindings) {
      std::cout << "binding " << binding.name.UTF8String << " type=" << binding.type
          << " access=" << binding.access << " index=" << binding.index << std::endl;
      if (binding.type != MTLBindingTypeBuffer || binding.index != 0) continue;
      id<MTLBufferBinding> b = (id<MTLBufferBinding>)binding;
      auto pointer = b.bufferPointerType;
      std::cout << " bufferType=" << b.bufferDataType << " struct=" << bool(b.bufferStructType)
          << " pointer=" << bool(pointer) << " elementAB=" << pointer.elementIsArgumentBuffer
          << " pointerElementType=" << pointer.elementType << " access=" << pointer.access << std::endl;
      auto structure = b.bufferStructType ?: pointer.elementStructType;
      for (MTLStructMember *m in structure.members) {
        auto a = m.arrayType;
        auto p = a.elementPointerType ?: m.pointerType;
        std::cout << " member " << m.name.UTF8String << " type=" << m.dataType
            << " argumentIndex=" << m.argumentIndex << " offset=" << m.offset
            << " arrayType=" << bool(a) << " elementType=" << a.elementType
            << " count=" << a.arrayLength << " byteStride=" << a.stride
            << " argumentStride=" << a.argumentIndexStride << " pointer=" << bool(p)
            << " pointerAccess=" << p.access << " alignment=" << p.alignment
            << " dataSize=" << p.dataSize << " elementAB=" << p.elementIsArgumentBuffer
            << std::endl;
        auto nested = m.structType;
        std::cout << "  nested members=" << nested.members.count << std::endl;
        for (NSUInteger i = 0; i < nested.members.count; ++i) {
          if (i >= 2 && i + 1 < nested.members.count) continue;
          MTLStructMember *child = nested.members[i];
          auto cp = child.pointerType;
          auto ca = child.arrayType;
          auto ep = ca.elementPointerType;
          std::cout << "  child " << child.name.UTF8String << " type=" << child.dataType
              << " argumentIndex=" << child.argumentIndex << " offset=" << child.offset
              << " pointer=" << bool(cp) << " array=" << bool(ca)
              << " pointerAccess=" << cp.access << " alignment=" << cp.alignment
              << " dataSize=" << cp.dataSize << " elementAB=" << cp.elementIsArgumentBuffer
              << " arrayElement=" << ca.elementType << " arrayCount=" << ca.arrayLength
              << " argumentStride=" << ca.argumentIndexStride << " elementPointer=" << bool(ep)
              << " elementAccess=" << ep.access << " elementAlignment=" << ep.alignment
              << " elementDataSize=" << ep.dataSize << " elementAB=" << ep.elementIsArgumentBuffer
              << std::endl;
        }
      }
    }
    std::cout << "CPU reflection only; no GPU commands submitted" << std::endl;
  }
}
