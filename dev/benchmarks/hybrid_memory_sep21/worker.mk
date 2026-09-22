BUILD ?= build/hybrid-q4-i8-fixed-r4-sep21-v2
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(BUILD)/source/runtime -mmacosx-version-min=27.0
CHANGED_NAMES := FlashMoE FlashMoEBlocked FlashExpertDenseCache FlashInt8ExpertStore FlashFloatDenseCache FlashDenseCache FlashForward FlashWorker
CHANGED := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(CHANGED_NAMES)))
BULK := $(BUILD)/host/Prefill4kQSABulk.o $(BUILD)/host/Prefill4kQSACoalesced.o
include $(BUILD)/link-inputs.mk
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
NONWORKER := $(filter-out $(BUILD)/host/FlashWorker.o,$(CHANGED)) $(BULK) $(REUSED)
AIR_HEADERS := $(shell rg --files $(BUILD)/source/runtime -g '*.h' -g '*.hpp')
.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash $(BUILD)/splash.metallib $(BUILD)/prefill4k-attribution $(BUILD)/policy-cpu

$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.cpp $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.mm $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/Prefill4kQSABulk.o: $(BUILD)/source/dev/benchmarks/prefill4k_attention/bulk.cpp $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/Prefill4kQSACoalesced.o: $(BUILD)/source/dev/benchmarks/prefill4k_attention/coalesced.cpp $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/splash-flash: $(CHANGED) $(BULK) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/prefill4k-attribution: $(BUILD)/source/dev/benchmarks/prefill4k_attribution.mm $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/policy-cpu: $(BUILD)/source/dev/benchmarks/moe_pointwise_sep21/policy_cpu.cpp $(BUILD)/source/dev/benchmarks/moe_pointwise_sep21/bridge.hpp $(BUILD)/overlay-manifest.json
	$(CXX) $(FLAGS) $< -o $@
$(BUILD)/pointwise.air: $(BUILD)/source/dev/benchmarks/moe_pointwise_sep21/candidate.metal $(BUILD)/overlay-manifest.json $(AIR_HEADERS)
	$(METAL) $(METALFLAGS) -Dflash_moe_blocked_poison_excluded_routes=private_moe_pointwise_unused_poison_baseline -c $< -o $@
$(BUILD)/bulk-attention.air: $(BUILD)/source/dev/benchmarks/prefill4k_attention/bulk_attention_sg8.metal $(BUILD)/overlay-manifest.json $(AIR_HEADERS)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/dense-prefill.air: $(BUILD)/source/runtime/metal/kernels/shared/flash_dense_cache_prefill.metal $(BUILD)/overlay-manifest.json $(AIR_HEADERS)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/splash.metallib: $(BUILD)/pointwise.air $(BUILD)/bulk-attention.air $(BUILD)/dense-prefill.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
cpu-self-test: $(BUILD)/policy-cpu $(BUILD)/splash-flash
	$(BUILD)/policy-cpu
	$(BUILD)/policy-cpu --freeze0
	$(BUILD)/policy-cpu --freeze1
	$(BUILD)/splash-flash --cpu-self-test
-include $(CHANGED:.o=.d) $(BULK:.o=.d)
