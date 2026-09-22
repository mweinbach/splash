BUILD ?= build/prefill4k-int8tiles-sweep
INITIAL ?= build/prefill4k-int8tiles
HOST ?= build/prefill4k-wide-fullcache
BASE ?= build/flash-next
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(HOST)/source/runtime -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
FLASH := $(filter-out $(HOST)/host/FlashWorker.o,$(wildcard $(HOST)/host/*.o))
CORE := $(BASE)/engine/metal/MetalBackend.o $(BASE)/engine/metal/DeviceCapabilities.o $(BASE)/engine/engine/Protocol.o $(BASE)/engine/engine/MemoryGovernor.o
.PHONY: all cpu-self-test
all: $(BUILD)/oracle $(BUILD)/splash.metallib
$(BUILD)/generated.stamp: dev/benchmarks/prefill4k_int8tiles/sweep_generate.py $(INITIAL)/oracle.mm dev/benchmarks/prefill4k_int8tiles/bridge.hpp
	mkdir -p $(BUILD)
	.venv/bin/python dev/benchmarks/prefill4k_int8tiles/sweep_generate.py $(BUILD)
	touch $@
$(BUILD)/oracle.o: $(BUILD)/generated.stamp
	$(CXX) $(FLAGS) -c $(BUILD)/oracle.mm -o $@
$(BUILD)/oracle: $(BUILD)/oracle.o $(FLASH) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/splash.metallib: $(INITIAL)/splash.metallib
	mkdir -p $(BUILD)
	cp $< $@
cpu-self-test: $(BUILD)/oracle
	$(BUILD)/oracle --cpu-self-test
