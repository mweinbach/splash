BUILD ?= build/moe-pointwise-sep21-worker-v1
BASE ?= build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1
COREBASE ?= build/flash-next
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(BUILD)/source/runtime -Iruntime -mmacosx-version-min=27.0
CHANGED_NAMES := FlashMoE FlashMoEBlocked FlashInt8ExpertStore FlashExpertDenseCache FlashForward FlashWorker
CHANGED := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(CHANGED_NAMES)))
REUSED := $(filter-out $(addprefix $(BASE)/host/,$(addsuffix .o,$(CHANGED_NAMES))),$(wildcard $(BASE)/host/*.o))
CORE := $(COREBASE)/engine/metal/MetalBackend.o $(COREBASE)/engine/metal/DeviceCapabilities.o $(COREBASE)/engine/engine/Protocol.o $(COREBASE)/engine/engine/MemoryGovernor.o
EXCLUDED := $(addprefix $(COREBASE)/metal/shared/,$(addsuffix .air,flash_gdn flash_gdn_fused flash_gdn_staged flash_int8_expert_store flash_qsa_bulk))
AIRS := $(filter-out $(EXCLUDED),$(wildcard $(COREBASE)/metal/*/*.air)) $(wildcard $(BASE)/metal/*.air)
include $(BUILD)/link-inputs.mk
NONWORKER := $(filter-out $(BUILD)/host/FlashWorker.o,$(CHANGED)) $(REUSED)
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash $(BUILD)/prefill4k-attribution $(BUILD)/splash.metallib $(BUILD)/policy-cpu
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.cpp $(BUILD)/source/dev/benchmarks/moe_pointwise_sep21/bridge.hpp $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.mm $(BUILD)/source/dev/benchmarks/moe_pointwise_sep21/bridge.hpp $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/splash-flash: $(CHANGED) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/prefill4k-attribution: $(BUILD)/source/dev/benchmarks/prefill4k_attribution.mm $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/policy-cpu: $(BUILD)/source/dev/benchmarks/moe_pointwise_sep21/policy_cpu.cpp $(BUILD)/source/dev/benchmarks/moe_pointwise_sep21/bridge.hpp $(BUILD)/overlay-manifest.json
	$(CXX) $(FLAGS) $< -o $@
$(BUILD)/pointwise.air: $(BUILD)/source/dev/benchmarks/moe_pointwise_sep21/candidate.metal $(BUILD)/overlay-manifest.json $(addprefix $(BUILD)/source/runtime/metal/abi/,$(addsuffix .h,FlashMoE FlashMoEBlocked FlashMoEFused FlashMoEBuckets))
	$(METAL) $(METALFLAGS) -Dflash_moe_blocked_poison_excluded_routes=private_moe_pointwise_unused_poison_baseline -c $< -o $@
$(BUILD)/splash.metallib: $(BUILD)/pointwise.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
cpu-self-test: $(BUILD)/policy-cpu $(BUILD)/splash-flash
	$(BUILD)/policy-cpu
	$(BUILD)/policy-cpu --freeze0
	$(BUILD)/policy-cpu --freeze1
	$(BUILD)/splash-flash --cpu-self-test
-include $(CHANGED:.o=.d)
