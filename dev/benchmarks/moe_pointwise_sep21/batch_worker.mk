BUILD ?= build/prefill4k-batch-pointwise-fma-sep21-v1
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
CHANGED_NAMES := FlashExpertDenseCache FlashForward FlashInt8ExpertStore FlashMoE FlashMoEBlocked FlashWorker FlashGDNStaged FlashBatchPrefill
CHANGED := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(CHANGED_NAMES)))
include $(BUILD)/link-inputs.mk
NONWORKER := $(filter-out $(BUILD)/host/FlashWorker.o,$(CHANGED)) $(REUSED)
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash $(BUILD)/splash.metallib $(BUILD)/prefill4k-attribution $(BUILD)/pointwise-policy-cpu $(BUILD)/fma-policy-cpu $(BUILD)/teacher-policy-cpu $(BUILD)/memory-cpu $(BUILD)/batch-teacher-oracle
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.cpp $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.mm $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/splash-flash: $(CHANGED) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/splash.metallib: $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
$(BUILD)/prefill4k-attribution: $(BUILD)/source/dev/benchmarks/prefill4k_attribution.mm $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/pointwise-policy-cpu: $(BUILD)/source/dev/benchmarks/moe_pointwise_sep21/policy_cpu.cpp $(BUILD)/overlay-manifest.json
	$(CXX) $(FLAGS) $< -o $@
$(BUILD)/fma-policy-cpu: $(BUILD)/source/dev/benchmarks/gdn_chunk_sep21/worker_policy_cpu.cpp $(BUILD)/overlay-manifest.json
	$(CXX) $(FLAGS) $< -o $@
$(BUILD)/teacher-policy-cpu: $(BUILD)/source/dev/benchmarks/prefill_batch_teacher_sep21/policy_cpu.cpp $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/memory-cpu: $(BUILD)/source/dev/benchmarks/prefill4k_wide_memory.mm $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/batch-teacher-oracle: $(BUILD)/source/dev/benchmarks/prefill_batch_teacher_sep21/oracle.mm $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
cpu-self-test: $(BUILD)/pointwise-policy-cpu $(BUILD)/fma-policy-cpu $(BUILD)/teacher-policy-cpu $(BUILD)/splash-flash
	$(BUILD)/pointwise-policy-cpu
	$(BUILD)/pointwise-policy-cpu --freeze0
	$(BUILD)/pointwise-policy-cpu --freeze1
	$(BUILD)/fma-policy-cpu
	$(BUILD)/fma-policy-cpu --freeze0
	$(BUILD)/fma-policy-cpu --freeze1
	$(BUILD)/teacher-policy-cpu
	$(BUILD)/splash-flash --cpu-self-test
-include $(CHANGED:.o=.d)
