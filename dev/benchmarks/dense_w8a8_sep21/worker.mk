BUILD ?= build/dense-w8a8-sep21-worker-v3
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
DIR := dev/benchmarks/dense_w8a8_sep21
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(BUILD)/source/runtime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
OWN := $(BUILD)/host/FlashForward.o $(BUILD)/host/FlashWorker.o $(BUILD)/host/worker_cache.o
include $(BUILD)/link-inputs.mk
NONWORKER := $(filter-out $(BUILD)/host/FlashWorker.o,$(OWN)) $(REUSED)
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
PRIVATE_HEADERS := $(addprefix $(BUILD)/source/$(DIR)/,worker_bridge.hpp worker_cache.hpp quantization.hpp abi.hpp)

.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash $(BUILD)/splash.metallib $(BUILD)/prefill4k-attribution $(BUILD)/policy-cpu $(BUILD)/cache-policy-cpu

$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.cpp $(PRIVATE_HEADERS) $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@

$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.mm $(PRIVATE_HEADERS) $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@

$(BUILD)/host/worker_cache.o: $(BUILD)/source/$(DIR)/worker_cache.cpp $(PRIVATE_HEADERS) $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@

$(BUILD)/splash-flash: $(OWN) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@

$(BUILD)/prefill4k-attribution: $(BUILD)/source/dev/benchmarks/prefill4k_attribution.mm $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) -MMD -MP $^ $(FRAMEWORKS) -o $@

$(BUILD)/policy-cpu: $(BUILD)/source/$(DIR)/worker_policy_cpu.cpp $(PRIVATE_HEADERS) $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) -MMD -MP $< $(NONWORKER) $(CORE) $(FRAMEWORKS) -o $@

$(BUILD)/cache-policy-cpu: $(BUILD)/source/$(DIR)/cache_policy_cpu.cpp $(PRIVATE_HEADERS)
	$(CXX) $(FLAGS) -MMD -MP $< -o $@

$(BUILD)/dense-w8a8.air: $(BUILD)/source/$(DIR)/candidate.metal $(BUILD)/source/$(DIR)/abi.hpp $(BUILD)/source/runtime/metal/abi/FlashDenseCache.h $(BUILD)/source/runtime/metal/abi/FlashAffine.h $(BUILD)/source/runtime/metal/kernels/common/flash_dense_traversal.h $(BUILD)/overlay-manifest.json
	$(METAL) $(METALFLAGS) -c $< -o $@

$(BUILD)/splash.metallib: $(BUILD)/dense-w8a8.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@

cpu-self-test: all
	$(BUILD)/policy-cpu
	$(BUILD)/policy-cpu --freeze0
	$(BUILD)/policy-cpu --freeze1
	$(BUILD)/policy-cpu --invalid
	$(BUILD)/cache-policy-cpu
	$(BUILD)/splash-flash --cpu-self-test

-include $(OWN:.o=.d) $(BUILD)/policy-cpu.d $(BUILD)/cache-policy-cpu.d $(BUILD)/prefill4k-attribution.d
