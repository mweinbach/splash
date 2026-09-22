BUILD ?= build/prefill-moe-sep21-native-m16
HOST ?= build/prefill4k-wide-fullcache
BASE ?= build/flash-next
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -I$(HOST)/source/runtime -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
FLASH := $(filter-out $(HOST)/host/FlashWorker.o,$(wildcard $(HOST)/host/*.o))
CORE := $(BASE)/engine/metal/MetalBackend.o $(BASE)/engine/metal/DeviceCapabilities.o $(BASE)/engine/engine/Protocol.o $(BASE)/engine/engine/MemoryGovernor.o
.PHONY: all cpu-self-test
all: $(BUILD)/oracle $(BUILD)/splash.metallib
$(BUILD)/oracle.mm: dev/benchmarks/prefill_moe_sep21/native_m16/generate.py dev/benchmarks/prefill_moe_sep21/native_m64/generate.py build/prefill4k-int8tiles/oracle.mm
	mkdir -p $(BUILD)
	.venv/bin/python $< $(BUILD)
$(BUILD)/oracle.o: $(BUILD)/oracle.mm dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp
	$(CXX) $(FLAGS) -c $< -o $@
$(BUILD)/oracle: $(BUILD)/oracle.o $(FLASH) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/splash.metallib: $(HOST)/splash.metallib
	mkdir -p $(BUILD)
	cp $< $@
cpu-self-test: all
	$(BUILD)/oracle --cpu-self-test
