BUILD ?= build/prefill-moe-sep21-one-layer
HOST ?= build/prefill4k-wide-fullcache
BASE ?= build/flash-next
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -I$(HOST)/source/runtime -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -I. -mmacosx-version-min=27.0
FLASH := $(filter-out $(HOST)/host/FlashWorker.o,$(wildcard $(HOST)/host/*.o))
CORE := $(BASE)/engine/metal/MetalBackend.o $(BASE)/engine/metal/DeviceCapabilities.o $(BASE)/engine/engine/Protocol.o $(BASE)/engine/engine/MemoryGovernor.o
AIRS := $(filter-out $(BASE)/metal/shared/flash_int8_expert_store.air,$(wildcard $(BASE)/metal/*/*.air)) $(HOST)/metal/flash_int8_expert_store.air
PRIVATE_AIRS := build/prefill-moe-sep21/memory.air build/prefill-moe-sep21/register.air

.PHONY: all cpu-self-test
all: $(BUILD)/oracle $(BUILD)/splash.metallib

$(BUILD)/oracle.mm: dev/benchmarks/prefill_moe_sep21/one_layer/generate.py dev/benchmarks/prefill_moe_sep21/native_m64/generate.py dev/benchmarks/prefill_moe_sep21/bridge.hpp dev/benchmarks/prefill4k_int8columns/precision.hpp build/prefill4k-int8tiles/oracle.mm
	mkdir -p $(BUILD)
	.venv/bin/python $< $(BUILD)

$(BUILD)/oracle.o: $(BUILD)/oracle.mm dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp
	$(CXX) $(FLAGS) -c $< -o $@

$(BUILD)/oracle: $(BUILD)/oracle.o $(FLASH) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@

build/prefill-moe-sep21/memory.air: dev/benchmarks/prefill_moe_sep21/memory.metal
	mkdir -p $(@D)
	$(METAL) $(METALFLAGS) -c $< -o $@

build/prefill-moe-sep21/register.air: dev/benchmarks/prefill_moe_sep21/register.metal
	mkdir -p $(@D)
	$(METAL) $(METALFLAGS) -c $< -o $@

$(BUILD)/splash.metallib: $(PRIVATE_AIRS) $(AIRS)
	mkdir -p $(BUILD)
	xcrun -sdk macosx metallib $^ -o $@

cpu-self-test: all
	$(BUILD)/oracle --cpu-self-test
