BUILD ?= build/prefill4k-bf16cache
BASE ?= build/flash-next
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0
FLASH := $(filter-out $(BASE)/flash/FlashWorker.o,$(wildcard $(BASE)/flash/*.o))
CORE := $(BASE)/engine/metal/MetalBackend.o $(BASE)/engine/metal/DeviceCapabilities.o $(BASE)/engine/engine/Protocol.o $(BASE)/engine/engine/MemoryGovernor.o
AIRS := $(wildcard $(BASE)/metal/*/*.air)
.PHONY: all cpu-self-test
all: $(BUILD)/oracle $(BUILD)/splash.metallib
$(BUILD)/generated.stamp: dev/benchmarks/prefill4k_bf16cache/generate.py dev/benchmarks/prefill4k_bf16cache/bridge.hpp runtime/metal/kernels/shared/flash_int8_expert_store.metal dev/benchmarks/flash_expert_dense_cache_oracle.mm
	mkdir -p $(BUILD)
	.venv/bin/python dev/benchmarks/prefill4k_bf16cache/generate.py $(BUILD)
	touch $@
$(BUILD)/oracle.o: $(BUILD)/generated.stamp
	$(CXX) $(FLAGS) -c $(BUILD)/oracle.mm -o $@
$(BUILD)/oracle: $(BUILD)/oracle.o $(FLASH) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/candidate.air: $(BUILD)/generated.stamp
	$(METAL) $(METALFLAGS) -c $(BUILD)/candidate.metal -o $@
$(BUILD)/splash.metallib: $(BUILD)/candidate.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
cpu-self-test: $(BUILD)/oracle
	$(BUILD)/oracle --cpu-self-test
