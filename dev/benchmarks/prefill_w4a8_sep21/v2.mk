# Private prelaunch fixes: no production/public metadata objects rebuilt.
W4_V2 ?= build/prefill-w4a8-sep21-v2
W4_V2_SOURCE := $(W4_V2)/source
W4_V2_FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -I$(W4_V2_SOURCE)/private-runtime -Iruntime -I. -Idev/benchmarks -I$(W4_V2_SOURCE) -fobjc-arc -mmacosx-version-min=27.0
W4_V2_OBJECTS := $(addprefix build/flash-next/flash/,FlashMoE.o FlashMoEBlocked.o FlashMoEBuckets.o) $(addprefix build/flash-next/engine/metal/,MetalBackend.o DeviceCapabilities.o) build/flash-next/engine/engine/MemoryGovernor.o $(W4_V2)/metadata.o
.PHONY: all cpu-self-test
all: $(W4_V2)/oracle
$(W4_V2)/metadata.o: $(W4_V2_SOURCE)/private-runtime/flash/FlashInt8ExpertStoreMetadata.mm $(W4_V2_SOURCE)/private-runtime/flash/FlashInt8ExpertStoreMetadata.hpp
	xcrun -sdk macosx clang++ $(W4_V2_FLAGS) -MMD -MP -c $< -o $@
$(W4_V2)/oracle.o: $(W4_V2_SOURCE)/oracle.mm $(W4_V2_SOURCE)/loader.hpp $(W4_V2_SOURCE)/precision.hpp $(W4_V2_SOURCE)/abi.h
	xcrun -sdk macosx clang++ $(W4_V2_FLAGS) -MMD -MP -c $< -o $@
$(W4_V2)/oracle: $(W4_V2)/oracle.o $(W4_V2_OBJECTS)
	xcrun -sdk macosx clang++ $(W4_V2_FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
cpu-self-test: $(W4_V2)/oracle
	$(W4_V2)/oracle --cpu-self-test
-include $(W4_V2)/oracle.d $(W4_V2)/metadata.d
