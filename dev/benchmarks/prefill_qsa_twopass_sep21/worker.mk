BUILD ?= build/prefill-qsa-twopass-sep21-worker-v1
CXX := xcrun -sdk macosx clang++
DIR := dev/benchmarks/prefill_qsa_twopass_sep21
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention -I$(BUILD)/source/$(DIR) -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
OWN := $(BUILD)/host/FlashForward.o $(BUILD)/host/FlashWorker.o $(BUILD)/host/twopass.o
include $(BUILD)/link-inputs.mk
NONWORKER := $(BUILD)/host/FlashForward.o $(BUILD)/host/twopass.o $(REUSED)
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
PRIVATE_HEADERS := $(BUILD)/source/$(DIR)/worker_bridge.hpp $(BUILD)/source/$(DIR)/twopass.hpp
.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash $(BUILD)/splash.metallib $(BUILD)/prefill4k-attribution $(BUILD)/policy-cpu
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.cpp $(PRIVATE_HEADERS) $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.mm $(PRIVATE_HEADERS) $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/twopass.o: $(BUILD)/source/$(DIR)/twopass.cpp $(PRIVATE_HEADERS)
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/splash-flash: $(OWN) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/prefill4k-attribution: $(BUILD)/source/dev/benchmarks/prefill4k_attribution.mm $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) -MMD -MP $< $(NONWORKER) $(CORE) $(FRAMEWORKS) -o $@
$(BUILD)/policy-cpu: $(BUILD)/source/$(DIR)/worker_policy_cpu.cpp $(PRIVATE_HEADERS) $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) -MMD -MP $< $(NONWORKER) $(CORE) $(FRAMEWORKS) -o $@
$(BUILD)/splash.metallib: $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
-include $(OWN:.o=.d) $(BUILD)/policy-cpu.d $(BUILD)/prefill4k-attribution.d
