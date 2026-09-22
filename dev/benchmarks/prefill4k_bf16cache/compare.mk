# Standalone private comparator. Reuses ready parent BF16 AIR and private
# combined-wide/fullcache host objects; no production build/default mutation.
BUILD ?= build/prefill4k-bf16-int8-compare
PARENT ?= build/prefill4k-bf16cache
HOST ?= build/prefill4k-wide-fullcache
BASE ?= build/flash-next
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(HOST)/source/runtime -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
FLASH := $(filter-out $(HOST)/host/FlashWorker.o,$(wildcard $(HOST)/host/*.o))
CORE := $(BASE)/engine/metal/MetalBackend.o $(BASE)/engine/metal/DeviceCapabilities.o $(BASE)/engine/engine/Protocol.o $(BASE)/engine/engine/MemoryGovernor.o
AIRS := $(filter-out $(BASE)/metal/shared/flash_int8_expert_store.air,$(wildcard $(BASE)/metal/*/*.air)) $(HOST)/metal/flash_int8_expert_store.air
.PHONY: all cpu-self-test
all: $(BUILD)/oracle $(BUILD)/splash.metallib
$(BUILD)/generated.stamp: dev/benchmarks/prefill4k_bf16cache/compare_generate.py $(PARENT)/oracle.mm $(PARENT)/candidate.air dev/benchmarks/prefill4k_bf16cache/bridge.hpp
	mkdir -p $(BUILD)
	.venv/bin/python dev/benchmarks/prefill4k_bf16cache/compare_generate.py generate $(BUILD) --parent-build $(PARENT)
	touch $@
$(BUILD)/oracle.o: $(BUILD)/generated.stamp
	$(CXX) $(FLAGS) -MMD -MP -c $(BUILD)/oracle.mm -o $@
$(BUILD)/oracle: $(BUILD)/oracle.o $(FLASH) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/splash.metallib: $(PARENT)/candidate.air $(AIRS)
	mkdir -p $(BUILD)
	xcrun -sdk macosx metallib $^ -o $@
cpu-self-test: all
	$(BUILD)/oracle --cpu-self-test
-include $(BUILD)/oracle.d
