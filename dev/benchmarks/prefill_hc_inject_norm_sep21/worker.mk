BUILD ?= build/prefill-hc-inject-norm-sep21-worker-v3
CXX := xcrun -sdk macosx clang++
DIR := dev/benchmarks/prefill_hc_inject_norm_sep21
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
OWN := $(BUILD)/host/FlashForward.o $(BUILD)/host/FlashWorker.o
include $(BUILD)/link-inputs.mk
NONWORKER := $(BUILD)/host/FlashForward.o $(REUSED)
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
PRIVATE_HEADERS := $(BUILD)/source/$(DIR)/bridge.hpp $(BUILD)/source/$(DIR)/worker_bridge.hpp
.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash $(BUILD)/splash.metallib $(BUILD)/prefill4k-attribution $(BUILD)/policy-cpu
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.cpp $(PRIVATE_HEADERS) $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.mm $(PRIVATE_HEADERS) $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/splash-flash: $(OWN) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/prefill4k-attribution: $(BUILD)/source/dev/benchmarks/prefill4k_attribution.mm $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) -MMD -MP $^ $(FRAMEWORKS) -o $@
$(BUILD)/policy-cpu: $(BUILD)/source/$(DIR)/worker_policy_cpu.cpp $(PRIVATE_HEADERS) $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) -MMD -MP $< $(NONWORKER) $(CORE) $(FRAMEWORKS) -o $@
$(BUILD)/splash.metallib: $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
cpu-self-test: all
	$(BUILD)/policy-cpu
	$(BUILD)/policy-cpu --freeze0
	$(BUILD)/policy-cpu --freeze1
	$(BUILD)/splash-flash --cpu-self-test
-include $(OWN:.o=.d) $(BUILD)/policy-cpu.d $(BUILD)/prefill4k-attribution.d
