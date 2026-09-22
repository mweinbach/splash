ROOT_HOST ?= build/prefill4k-qsa-bulk-gathered-bf16-sep21-v3
BUILD ?= build/sep21-bf16-narrow-role-screen-v2
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -ffp-contract=off -fno-fast-math -fobjc-arc -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -I$(ROOT_HOST)/source/runtime -Iruntime -I. -Idev/benchmarks/decode_bf16_narrow_sep21
HOST_OBJECTS := $(filter-out $(ROOT_HOST)/host/FlashWorker.o,$(wildcard $(ROOT_HOST)/host/*.o))
CORE_OBJECTS := build/engine/metal/MetalBackend.o build/engine/metal/DeviceCapabilities.o build/engine/engine/Protocol.o build/engine/engine/MemoryGovernor.o
PARENT_AIRS := $(wildcard build/prefill4k-qsa-bulk-gathered-bf16-sep21-v2/metal/*.air) $(filter-out build/metal/shared/flash_qsa_bulk.air build/metal/shared/flash_int8_expert_store.air build/metal/shared/flash_gdn.air build/metal/shared/flash_gdn_fused.air build/metal/shared/flash_gdn_staged.air,$(wildcard build/metal/*/*.air))
.PHONY: all cpu
all: $(BUILD)/oracle $(BUILD)/splash.metallib
$(BUILD)/oracle: dev/benchmarks/decode_bf16_narrow_sep21/oracle.mm dev/benchmarks/decode_bf16_narrow_sep21/KernelParams.hpp $(HOST_OBJECTS) $(CORE_OBJECTS)
	@mkdir -p $(BUILD)
	$(CXX) $(FLAGS) $< $(HOST_OBJECTS) $(CORE_OBJECTS) -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/candidate.air: dev/benchmarks/decode_bf16_narrow_sep21/kernels.metal dev/benchmarks/decode_bf16_narrow_sep21/KernelParams.hpp
	@mkdir -p $(BUILD)
	xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -I$(ROOT_HOST)/source/runtime -Iruntime -Idev/benchmarks/decode_bf16_narrow_sep21 -c $< -o $@
$(BUILD)/splash.metallib: $(BUILD)/candidate.air $(PARENT_AIRS)
	xcrun -sdk macosx metallib $^ -o $@
cpu: all
	$(BUILD)/oracle --cpu-self-test
