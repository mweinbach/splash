BUILD ?= build/prefill-moe-sep21-worker-v2
BASE ?= build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1
COREBASE ?= build/flash-next
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(BUILD)/source/runtime -Iruntime -I. -mmacosx-version-min=27.0
REUSED := $(filter-out $(BASE)/host/FlashInt8ExpertStore.o $(BASE)/host/FlashMoEBlocked.o $(BASE)/host/FlashWorker.o,$(wildcard $(BASE)/host/*.o))
NONWORKER := $(REUSED)
CORE := $(COREBASE)/engine/metal/MetalBackend.o $(COREBASE)/engine/metal/DeviceCapabilities.o $(COREBASE)/engine/engine/Protocol.o $(COREBASE)/engine/engine/MemoryGovernor.o
EXCLUDED := $(addprefix $(COREBASE)/metal/shared/,$(addsuffix .air,flash_gdn flash_gdn_fused flash_gdn_staged flash_int8_expert_store flash_qsa_bulk))
AIRS := $(filter-out $(EXCLUDED),$(wildcard $(COREBASE)/metal/*/*.air)) $(wildcard $(BASE)/metal/*.air)
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash $(BUILD)/prefill4k-attribution $(BUILD)/splash.metallib
$(BUILD)/FlashInt8ExpertStore.o: $(BUILD)/source/runtime/flash/FlashInt8ExpertStore.mm $(BUILD)/source/dev/benchmarks/prefill_moe_sep21/bridge.hpp $(BUILD)/overlay-manifest.json
	$(CXX) $(FLAGS) -c $< -o $@
$(BUILD)/FlashMoEBlocked.o: $(BUILD)/source/runtime/flash/FlashMoEBlocked.cpp $(BUILD)/source/dev/benchmarks/prefill_moe_sep21/bridge.hpp $(BUILD)/overlay-manifest.json
	$(CXX) $(FLAGS) -c $< -o $@
$(BUILD)/FlashWorker.o: $(BUILD)/source/runtime/flash/FlashWorker.mm $(BUILD)/source/dev/benchmarks/prefill_moe_sep21/bridge.hpp $(BUILD)/overlay-manifest.json
	$(CXX) $(FLAGS) -c $< -o $@
$(BUILD)/splash-flash: $(BUILD)/FlashInt8ExpertStore.o $(BUILD)/FlashMoEBlocked.o $(BUILD)/FlashWorker.o $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/prefill4k-attribution: dev/benchmarks/prefill4k_attribution.mm $(BUILD)/FlashInt8ExpertStore.o $(BUILD)/FlashMoEBlocked.o $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/memory.air: $(BUILD)/source/dev/benchmarks/prefill_moe_sep21/memory.metal
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/register.air: $(BUILD)/source/dev/benchmarks/prefill_moe_sep21/register.metal
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/splash.metallib: $(BUILD)/memory.air $(BUILD)/register.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
cpu-self-test: $(BUILD)/splash-flash
	$(BUILD)/splash-flash --cpu-self-test
$(BUILD)/identity-cpu: dev/benchmarks/prefill_moe_sep21/identity_cpu.cpp $(BUILD)/source/dev/benchmarks/prefill_moe_sep21/bridge.hpp
	$(CXX) $(FLAGS) $< -o $@
