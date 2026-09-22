BUILD ?= build/prefill-moe-sep21-w8a8-pointwise-worker-v1
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(BUILD)/source/runtime -mmacosx-version-min=27.0
OWN_NAMES := FlashInt8ExpertStore FlashForward FlashWorker
OWN := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(OWN_NAMES)))
include $(BUILD)/link-inputs.mk
NONWORKER := $(filter-out $(BUILD)/host/FlashWorker.o,$(OWN)) $(REUSED)
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash $(BUILD)/prefill4k-attribution $(BUILD)/splash.metallib $(BUILD)/policy-cpu
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.cpp $(BUILD)/source/dev/benchmarks/prefill_moe_sep21/w8a8/worker.hpp $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.mm $(BUILD)/source/dev/benchmarks/prefill_moe_sep21/w8a8/worker.hpp $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/splash-flash: $(OWN) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/prefill4k-attribution: $(BUILD)/source/dev/benchmarks/prefill4k_attribution.mm $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/policy-cpu: $(BUILD)/source/dev/benchmarks/prefill_moe_sep21/w8a8/worker_policy_cpu.cpp $(BUILD)/source/dev/benchmarks/prefill_moe_sep21/w8a8/worker.hpp
	$(CXX) $(FLAGS) $< -o $@
$(BUILD)/w8a8.air: $(BUILD)/source/dev/benchmarks/prefill_moe_sep21/w8a8/candidate.metal $(BUILD)/overlay-manifest.json
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/splash.metallib: $(BUILD)/w8a8.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
cpu-self-test: $(BUILD)/policy-cpu $(BUILD)/splash-flash
	$(BUILD)/policy-cpu
	$(BUILD)/splash-flash --cpu-self-test
-include $(OWN:.o=.d)
