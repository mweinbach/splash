BUILD ?= build/gemv-decode-r1-pointwise-sg2tail-sep21-worker-v1
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(BUILD)/source -I$(BUILD)/source/runtime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
DIR := dev/benchmarks/gemv_decode_r1_worker_sep21
OWN_NAMES := FlashInt8ExpertStore FlashForward FlashWorker
OWN := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(OWN_NAMES)))
include $(BUILD)/link-inputs.mk
NONWORKER := $(filter-out $(BUILD)/host/FlashWorker.o,$(OWN)) $(REUSED)
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash $(BUILD)/splash.metallib $(BUILD)/policy-cpu $(BUILD)/prefill4k-attribution
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.cpp $(BUILD)/source/$(DIR)/bridge.hpp $(BUILD)/source/$(DIR)/source_identity.hpp $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.mm $(BUILD)/source/$(DIR)/bridge.hpp $(BUILD)/source/$(DIR)/source_identity.hpp $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/splash-flash: $(OWN) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/prefill4k-attribution: $(BUILD)/source/dev/benchmarks/prefill4k_attribution.mm $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/policy-cpu: $(BUILD)/source/$(DIR)/policy_cpu.cpp $(BUILD)/source/$(DIR)/bridge.hpp $(BUILD)/source/$(DIR)/source_identity.hpp
	$(CXX) $(FLAGS) $< -o $@
$(BUILD)/vector-r1.air: $(BUILD)/source/$(DIR)/candidate.metal $(BUILD)/source/$(DIR)/abi.hpp $(BUILD)/overlay-manifest.json
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/splash.metallib: $(BUILD)/vector-r1.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
cpu-self-test: all
	$(BUILD)/policy-cpu
	$(BUILD)/policy-cpu --freeze0
	$(BUILD)/policy-cpu --freeze1
	$(BUILD)/policy-cpu --missing
	$(BUILD)/policy-cpu --retry0
	$(BUILD)/policy-cpu --retry1
	$(BUILD)/splash-flash --cpu-self-test
-include $(OWN:.o=.d)
