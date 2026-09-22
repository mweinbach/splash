BUILD ?= build/prefill4k-int8tiles
BASE ?= build/prefill4k-wide-fullcache
COREBASE ?= build/flash-next
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(BASE)/source/runtime -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0
FLASH := $(filter-out $(BASE)/host/FlashWorker.o,$(wildcard $(BASE)/host/*.o))
CORE := $(COREBASE)/engine/metal/MetalBackend.o $(COREBASE)/engine/metal/DeviceCapabilities.o $(COREBASE)/engine/engine/Protocol.o $(COREBASE)/engine/engine/MemoryGovernor.o
AIRS := $(filter-out $(COREBASE)/metal/shared/flash_int8_expert_store.air,$(wildcard $(COREBASE)/metal/*/*.air)) $(BASE)/metal/flash_int8_expert_store.air
.PHONY: all cpu-self-test
all: $(BUILD)/oracle $(BUILD)/splash.metallib
$(BUILD)/generated.stamp: dev/benchmarks/prefill4k_int8tiles/generate.py dev/benchmarks/prefill4k_int8tiles/candidate.metal dev/benchmarks/prefill4k_int8tiles/bridge.hpp dev/benchmarks/flash_int8_expert_store_oracle.mm
	mkdir -p $(BUILD)
	.venv/bin/python dev/benchmarks/prefill4k_int8tiles/generate.py $(BUILD)
	touch $@
$(BUILD)/oracle.o: $(BUILD)/generated.stamp
	$(CXX) $(FLAGS) -c $(BUILD)/oracle.mm -o $@
$(BUILD)/oracle: $(BUILD)/oracle.o $(FLASH) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/prefill4k_int8tiles_hits.air: dev/benchmarks/prefill4k_int8tiles/candidate.metal
	mkdir -p $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/jobs.air: dev/benchmarks/prefill4k_int8tiles/jobs.metal
	mkdir -p $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/relaxed.air: $(BUILD)/generated.stamp
	$(METAL) $(METALFLAGS) -c $(BUILD)/relaxed.metal -o $@
$(BUILD)/splash.metallib: $(BUILD)/prefill4k_int8tiles_hits.air $(BUILD)/jobs.air $(BUILD)/relaxed.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
cpu-self-test: $(BUILD)/oracle
	$(BUILD)/oracle --cpu-self-test
