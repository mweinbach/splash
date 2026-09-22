BUILD ?= build/adaptive-expert-tail-sep21-worker-v1
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(BUILD)/source -I$(BUILD)/source/runtime -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
CHANGED_NAMES := FlashInt8ExpertStore FlashForward FlashWorker
CHANGED := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(CHANGED_NAMES)))
include $(BUILD)/link-inputs.mk
NONWORKER := $(filter-out $(BUILD)/host/FlashWorker.o,$(CHANGED)) $(REUSED)
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit

.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash $(BUILD)/splash.metallib $(BUILD)/prefill4k-attribution $(BUILD)/policy-cpu

$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.cpp $(BUILD)/source/dev/benchmarks/adaptive_expert_tail_sep21/worker_bridge.hpp $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@

$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.mm $(BUILD)/source/dev/benchmarks/adaptive_expert_tail_sep21/worker_bridge.hpp $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@

$(BUILD)/splash-flash: $(CHANGED) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@

$(BUILD)/prefill4k-attribution: $(BUILD)/source/dev/benchmarks/prefill4k_attribution.mm $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@

$(BUILD)/policy-cpu: $(BUILD)/source/dev/benchmarks/adaptive_expert_tail_sep21/worker_policy_cpu.cpp $(BUILD)/source/dev/benchmarks/adaptive_expert_tail_sep21/worker_bridge.hpp $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) $< $(NONWORKER) $(CORE) $(FRAMEWORKS) -o $@

$(BUILD)/adaptive.air: $(BUILD)/source/dev/benchmarks/adaptive_expert_tail_sep21/adaptive.metal $(BUILD)/overlay-manifest.json
	$(METAL) $(METALFLAGS) -c $< -o $@

$(BUILD)/splash.metallib: $(BUILD)/adaptive.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@

cpu-self-test: $(BUILD)/policy-cpu $(BUILD)/splash-flash
	$(BUILD)/policy-cpu
	$(BUILD)/policy-cpu --freeze0
	$(BUILD)/policy-cpu --freeze1
	$(BUILD)/policy-cpu --missing0
	$(BUILD)/policy-cpu --retry0
	$(BUILD)/policy-cpu --retry1
	$(BUILD)/splash-flash --cpu-self-test

-include $(CHANGED:.o=.d)
