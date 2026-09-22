BUILD ?= build/prefill4k-q4nax
HOST ?= build/prefill4k-wide-fullcache
BASE ?= build/flash-next
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(HOST)/source/runtime -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -I. -mmacosx-version-min=27.0
FLASH := $(filter-out $(HOST)/host/FlashWorker.o,$(wildcard $(HOST)/host/*.o))
CORE := $(BASE)/engine/metal/MetalBackend.o $(BASE)/engine/metal/DeviceCapabilities.o $(BASE)/engine/engine/Protocol.o $(BASE)/engine/engine/MemoryGovernor.o
AIRS := $(filter-out $(BASE)/metal/shared/flash_int8_expert_store.air,$(wildcard $(BASE)/metal/*/*.air)) $(HOST)/metal/flash_int8_expert_store.air
.PHONY: all cpu-self-test
all: $(BUILD)/oracle $(BUILD)/splash.metallib
$(BUILD)/generated.stamp: dev/benchmarks/prefill4k_q4nax/generate.py dev/benchmarks/prefill4k_q4nax/candidate.metal dev/benchmarks/prefill4k_q4coded/precision.hpp build/prefill4k-q4coded/oracle.mm
	mkdir -p $(BUILD)
	.venv/bin/python dev/benchmarks/prefill4k_q4nax/generate.py $(BUILD)
	touch $@
$(BUILD)/oracle.o: $(BUILD)/generated.stamp
	$(CXX) $(FLAGS) -c $(BUILD)/oracle.mm -o $@
$(BUILD)/oracle: $(BUILD)/oracle.o $(FLASH) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/candidate.air: dev/benchmarks/prefill4k_q4nax/candidate.metal
	mkdir -p $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/audit.air: $(BUILD)/generated.stamp
	$(METAL) $(METALFLAGS) -c $(BUILD)/audit.metal -o $@
$(BUILD)/splash.metallib: $(BUILD)/candidate.air $(BUILD)/audit.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
cpu-self-test: $(BUILD)/oracle
	$(BUILD)/oracle --cpu-self-test
