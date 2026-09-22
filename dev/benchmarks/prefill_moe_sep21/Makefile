BUILD ?= build/prefill-moe-sep21
HOST ?= build/prefill4k-wide-fullcache
BASE ?= build/flash-next
GATHERED ?= build/prefill4k-allrows-gathered-mpp-component-v3
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(HOST)/source/runtime -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -I. -mmacosx-version-min=27.0
FLASH := $(filter-out $(HOST)/host/FlashWorker.o,$(wildcard $(HOST)/host/*.o))
CORE := $(BASE)/engine/metal/MetalBackend.o $(BASE)/engine/metal/DeviceCapabilities.o $(BASE)/engine/engine/Protocol.o $(BASE)/engine/engine/MemoryGovernor.o
AIRS := $(filter-out $(BASE)/metal/shared/flash_int8_expert_store.air,$(wildcard $(BASE)/metal/*/*.air)) $(HOST)/metal/flash_int8_expert_store.air
.PHONY: all cpu-self-test gathered
all: $(BUILD)/oracle $(BUILD)/splash.metallib
$(BUILD)/generated.stamp: dev/benchmarks/prefill_moe_sep21/generate_oracle.py dev/benchmarks/prefill_moe_sep21/bridge.hpp build/prefill4k-int8tiles/oracle.mm dev/benchmarks/prefill4k_int8columns/precision.hpp
	mkdir -p $(BUILD)
	.venv/bin/python dev/benchmarks/prefill_moe_sep21/generate_oracle.py $(BUILD)
	touch $@
$(BUILD)/oracle.o: $(BUILD)/generated.stamp
	$(CXX) $(FLAGS) -c $(BUILD)/oracle.mm -o $@
$(BUILD)/oracle: $(BUILD)/oracle.o $(FLASH) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/memory.air: dev/benchmarks/prefill_moe_sep21/memory.metal
	mkdir -p $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/register.air: dev/benchmarks/prefill_moe_sep21/register.metal
	mkdir -p $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/splash.metallib: $(BUILD)/memory.air $(BUILD)/register.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
$(BUILD)/gathered_sg%.air: dev/benchmarks/prefill_moe_sep21/gathered_sg%.metal
	mkdir -p $(BUILD)
	$(METAL) $(METALFLAGS) -I$(GATHERED)/source/runtime -c $< -o $@
gathered: $(BUILD)/gathered_sg1.air $(BUILD)/gathered_sg2.air
cpu-self-test: $(BUILD)/oracle
	$(BUILD)/oracle --cpu-self-test
