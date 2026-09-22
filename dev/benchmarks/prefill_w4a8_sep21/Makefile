# Private bounded one-layer build; no target creates a device or runs GPU work.
W4_BUILD ?= build/prefill-w4a8-sep21
W4_SOURCE := dev/benchmarks/prefill_w4a8_sep21
W4_FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -Iruntime -I. -Idev/benchmarks -I$(W4_SOURCE) -fobjc-arc -mmacosx-version-min=27.0
W4_METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -I$(W4_SOURCE) -mmacosx-version-min=27.0
W4_OBJECTS := $(addprefix build/flash-next/flash/,FlashMoE.o FlashMoEBlocked.o FlashMoEBuckets.o FlashInt8ExpertStoreMetadata.o) $(addprefix build/flash-next/engine/metal/,MetalBackend.o DeviceCapabilities.o) build/flash-next/engine/engine/MemoryGovernor.o
W4_AIRS := $(addprefix build/flash-next/metal/shared/,flash_moe.air flash_moe_blocked.air flash_moe_buckets.air flash_moe_direct_a.air flash_moe_q4x8.air flash_int8_expert_store.air) build/prefill-moe-sep21-w8a8/candidate.air

.PHONY: all cpu-self-test
all: $(W4_BUILD)/oracle $(W4_BUILD)/candidate.metallib
$(W4_BUILD):
	mkdir -p $@
$(W4_BUILD)/candidate.air: $(W4_SOURCE)/candidate.metal $(W4_SOURCE)/abi.h | $(W4_BUILD)
	xcrun -sdk macosx metal $(W4_METALFLAGS) -c $< -o $@
$(W4_BUILD)/candidate.metallib: $(W4_BUILD)/candidate.air $(W4_AIRS)
	xcrun -sdk macosx metallib $^ -o $@
$(W4_BUILD)/oracle.o: $(W4_SOURCE)/oracle.mm $(W4_SOURCE)/loader.hpp $(W4_SOURCE)/precision.hpp $(W4_SOURCE)/abi.h | $(W4_BUILD)
	xcrun -sdk macosx clang++ $(W4_FLAGS) -MMD -MP -c $< -o $@
$(W4_BUILD)/oracle: $(W4_BUILD)/oracle.o $(W4_OBJECTS)
	xcrun -sdk macosx clang++ $(W4_FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
cpu-self-test: $(W4_BUILD)/oracle
	$(W4_BUILD)/oracle --cpu-self-test
-include $(W4_BUILD)/oracle.d
