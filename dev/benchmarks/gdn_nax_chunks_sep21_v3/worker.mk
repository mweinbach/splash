BUILD ?= build/gdn-wy-dense-prefill-sep21-worker-v1
CXX := xcrun -sdk macosx clang++
DIR := dev/benchmarks/gdn_nax_chunks_sep21_v3
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(BUILD)/source/runtime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
OWN := $(BUILD)/host/FlashForward.o $(BUILD)/host/FlashWorker.o $(BUILD)/host/FlashBatchPrefill.o $(BUILD)/host/MetalBackend.o
include $(BUILD)/link-inputs.mk
NONWORKER := $(filter-out $(BUILD)/host/FlashWorker.o,$(OWN)) $(REUSED)
PRIVATE_HEADERS := $(BUILD)/source/$(DIR)/worker_bridge.hpp $(BUILD)/source/$(DIR)/worker_source_hashes.hpp
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit

.PHONY: all policy-checks
all: $(BUILD)/splash-flash $(BUILD)/splash.metallib $(BUILD)/prefill4k-attribution $(BUILD)/policy-cpu

$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.cpp $(PRIVATE_HEADERS) $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.mm $(PRIVATE_HEADERS) $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/MetalBackend.o: $(BUILD)/source/runtime/metal/MetalBackend.mm $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/splash-flash: $(OWN) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/prefill4k-attribution: $(BUILD)/source/dev/benchmarks/prefill4k_attribution.mm $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) -MMD -MP $(filter %.mm %.o,$^) $(FRAMEWORKS) -o $@
$(BUILD)/policy-cpu: $(BUILD)/source/$(DIR)/worker_policy_cpu.cpp $(PRIVATE_HEADERS) $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) -MMD -MP $< $(NONWORKER) $(CORE) $(FRAMEWORKS) -o $@
$(BUILD)/candidate.air: $(BUILD)/source/$(DIR)/candidate.metal $(BUILD)/source/runtime/metal/abi/FlashGDN.h $(BUILD)/overlay-manifest.json
	xcrun -sdk macosx metal $(METALFLAGS) -c $< -o $@
$(BUILD)/snapshot.air: $(BUILD)/source/$(DIR)/snapshot.metal $(BUILD)/source/runtime/metal/abi/FlashGDN.h $(BUILD)/overlay-manifest.json
	xcrun -sdk macosx metal $(METALFLAGS) -c $< -o $@
$(BUILD)/native-fallback.air: $(BUILD)/source/$(DIR)/native_fallback.metal $(BUILD)/source/runtime/metal/abi/FlashGDN.h $(BUILD)/overlay-manifest.json
	xcrun -sdk macosx metal $(METALFLAGS) -c $< -o $@
$(BUILD)/splash.metallib: $(BUILD)/candidate.air $(BUILD)/snapshot.air $(BUILD)/native-fallback.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@

policy-checks: $(BUILD)/policy-cpu
	$(BUILD)/policy-cpu
	$(BUILD)/policy-cpu --freeze0
	$(BUILD)/policy-cpu --freeze1
	$(BUILD)/policy-cpu --invalid
	$(BUILD)/policy-cpu --missing-staged

-include $(OWN:.o=.d) $(BUILD)/policy-cpu.d $(BUILD)/prefill4k-attribution.d
