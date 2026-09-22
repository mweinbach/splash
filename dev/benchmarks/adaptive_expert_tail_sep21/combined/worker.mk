BUILD ?= build/adaptive-expert-tail-sg2k128-fma-sep21-worker-v1
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(BUILD)/source/runtime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
OWN_NAMES := FlashInt8ExpertStore FlashForward FlashWorker FlashGDNStaged FlashBatchPrefill
OWN := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(OWN_NAMES)))
include $(BUILD)/link-inputs.mk
NONWORKER := $(filter-out $(BUILD)/host/FlashWorker.o,$(OWN)) $(REUSED)
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
DIR := dev/benchmarks/adaptive_expert_tail_sep21/combined

.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash $(BUILD)/splash.metallib $(BUILD)/prefill4k-attribution $(BUILD)/policy-cpu $(BUILD)/fma-policy-cpu $(BUILD)/inherited-sg2-policy-cpu

$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.cpp $(BUILD)/source/$(DIR)/worker_bridge.hpp $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@

$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.mm $(BUILD)/source/$(DIR)/worker_bridge.hpp $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@

$(BUILD)/splash-flash: $(OWN) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@

$(BUILD)/prefill4k-attribution: $(BUILD)/source/dev/benchmarks/prefill4k_attribution.mm $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@

$(BUILD)/policy-cpu: $(BUILD)/source/$(DIR)/worker_policy_cpu.cpp $(BUILD)/source/$(DIR)/worker_bridge.hpp $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) $< $(NONWORKER) $(CORE) $(FRAMEWORKS) -o $@

$(BUILD)/fma-policy-cpu: $(BUILD)/source/dev/benchmarks/gdn_chunk_sep21/worker_policy_cpu.cpp $(BUILD)/source/dev/benchmarks/gdn_chunk_sep21/worker_bridge.hpp
	$(CXX) $(FLAGS) $< -o $@

$(BUILD)/inherited-sg2-policy-cpu: $(BUILD)/source/dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker/policy_cpu.cpp $(BUILD)/source/dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker/policy.hpp
	$(CXX) $(FLAGS) $< -o $@

$(BUILD)/adaptive.air: $(BUILD)/source/$(DIR)/adaptive.metal $(BUILD)/overlay-manifest.json
	$(METAL) $(METALFLAGS) -c $< -o $@

$(BUILD)/splash.metallib: $(BUILD)/adaptive.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@

cpu-self-test: all
	$(BUILD)/policy-cpu
	$(BUILD)/policy-cpu --freeze00
	$(BUILD)/policy-cpu --freeze01
	$(BUILD)/policy-cpu --freeze70
	$(BUILD)/policy-cpu --freeze71
	$(BUILD)/policy-cpu --missing00
	$(BUILD)/policy-cpu --retry00
	$(BUILD)/policy-cpu --retry71
	$(BUILD)/policy-cpu --sg2-first
	$(BUILD)/policy-cpu --tail-first
	$(BUILD)/fma-policy-cpu --source $(BUILD)/source/dev/benchmarks/gdn_chunk_sep21/scalar_fma.metal
	$(BUILD)/fma-policy-cpu --freeze0
	$(BUILD)/fma-policy-cpu --freeze1
	env SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21=0 SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT=0 $(BUILD)/inherited-sg2-policy-cpu
	env SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21=0 SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT=7 $(BUILD)/inherited-sg2-policy-cpu
	$(BUILD)/splash-flash --cpu-self-test

-include $(OWN:.o=.d)
