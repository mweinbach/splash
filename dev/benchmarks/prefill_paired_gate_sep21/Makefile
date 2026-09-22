# Private one-layer build; all/CPU targets never create Metal objects or read payloads.
PAIR_BUILD ?= build/prefill-paired-gate-sep21
PAIR_SOURCE := dev/benchmarks/prefill_paired_gate_sep21
PAIR_FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -Ibuild/prefill-w4a8-sep21-v3/source/private-runtime -Iruntime -I. -Idev/benchmarks -I$(PAIR_SOURCE) -fobjc-arc -mmacosx-version-min=27.0
PAIR_METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -I. -I$(PAIR_SOURCE) -mmacosx-version-min=27.0
PAIR_OBJECTS := $(addprefix build/flash-next/flash/,FlashMoE.o FlashMoEBlocked.o FlashMoEBuckets.o) $(addprefix build/flash-next/engine/metal/,MetalBackend.o DeviceCapabilities.o) build/flash-next/engine/engine/MemoryGovernor.o build/prefill-w4a8-sep21-v3/metadata.o
PAIR_AIRS := $(addprefix build/flash-next/metal/shared/,flash_moe.air flash_moe_blocked.air flash_moe_buckets.air flash_moe_direct_a.air flash_moe_q4x8.air flash_int8_expert_store.air) build/prefill-moe-sep21/memory.air build/prefill-moe-sep21-w8a8/candidate.air
.PHONY: all cpu-self-test
all: $(PAIR_BUILD)/oracle $(PAIR_BUILD)/candidate.metallib
$(PAIR_BUILD)/candidate.air: $(PAIR_SOURCE)/candidate.metal $(PAIR_SOURCE)/abi.h
	xcrun -sdk macosx metal $(PAIR_METALFLAGS) -c $< -o $@
$(PAIR_BUILD)/control.air: $(PAIR_SOURCE)/control.metal
	xcrun -sdk macosx metal $(PAIR_METALFLAGS) -c $< -o $@
$(PAIR_BUILD)/candidate.metallib: $(PAIR_BUILD)/candidate.air $(PAIR_BUILD)/control.air $(PAIR_AIRS)
	xcrun -sdk macosx metallib $^ -o $@
$(PAIR_BUILD)/oracle.o: $(PAIR_SOURCE)/oracle.mm $(PAIR_SOURCE)/packing.hpp $(PAIR_SOURCE)/abi.h
	xcrun -sdk macosx clang++ $(PAIR_FLAGS) -MMD -MP -c $< -o $@
$(PAIR_BUILD)/oracle: $(PAIR_BUILD)/oracle.o $(PAIR_OBJECTS)
	xcrun -sdk macosx clang++ $(PAIR_FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
cpu-self-test: $(PAIR_BUILD)/oracle
	$(PAIR_BUILD)/oracle --cpu-self-test
-include $(PAIR_BUILD)/oracle.d
