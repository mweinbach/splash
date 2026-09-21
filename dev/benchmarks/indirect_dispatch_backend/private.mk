# Include after the production Makefile, only in an isolated private build.
# Every host object must use this header overlay: ComputeDispatch ABI differs.
PRIVATE_INDIRECT_ROOT := dev/benchmarks/indirect_dispatch_backend
PRIVATE_INDIRECT_FLAGS := -I$(PRIVATE_INDIRECT_ROOT)/include \
	-I$(PRIVATE_INDIRECT_ROOT)/include/metal -Iruntime -Iruntime/metal
override ENGINE_CXXFLAGS := $(PRIVATE_INDIRECT_FLAGS) -std=c++20 -O3 -Wall -Wextra -Werror \
	-mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
override ENGINE_OBJCXXFLAGS := $(ENGINE_CXXFLAGS) -fobjc-arc

# The cloned backend includes its private header directly. Reusing the normal
# .mm would find the production sibling header before any -I overlay.
$(ENGINE_METAL_RUNTIME_OBJECT): $(PRIVATE_INDIRECT_ROOT)/MetalBackend.mm \
	$(PRIVATE_INDIRECT_ROOT)/include/metal/MetalBackend.hpp \
	$(PRIVATE_INDIRECT_ROOT)/IndirectDispatchPolicy.hpp
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $(ENGINE_DEPFLAGS) \
		-c $(PRIVATE_INDIRECT_ROOT)/MetalBackend.mm -o $@
