BUILD ?= build/prefill4k-moe
BASE ?= build/flash-next
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -I. -I$(BUILD) -mmacosx-version-min=27.0
FLASH := $(filter-out $(BASE)/flash/FlashWorker.o,$(wildcard $(BASE)/flash/*.o))
CORE := $(BASE)/engine/metal/MetalBackend.o $(BASE)/engine/metal/DeviceCapabilities.o $(BASE)/engine/engine/Protocol.o $(BASE)/engine/engine/MemoryGovernor.o
AIRS := $(wildcard $(BASE)/metal/*/*.air)
.PHONY: all cpu-self-test
all: $(BUILD)/oracle $(BUILD)/splash.metallib
$(BUILD)/generated.stamp: dev/benchmarks/prefill4k_moe_generate.py dev/benchmarks/prefill4k_moe_expanded.h runtime/metal/kernels/shared/flash_int8_expert_store.metal runtime/metal/kernels/common/flash_moe_direct_a_common.h dev/benchmarks/flash_int8_expert_store_oracle.mm
	mkdir -p $(BUILD)
	.venv/bin/python dev/benchmarks/prefill4k_moe_generate.py $(BUILD)
	touch $@
$(BUILD)/oracle.o: $(BUILD)/generated.stamp
	$(CXX) $(FLAGS) -c $(BUILD)/oracle.mm -o $@
$(BUILD)/oracle: $(BUILD)/oracle.o $(FLASH) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/static.air: $(BUILD)/generated.stamp
	$(METAL) $(METALFLAGS) -c $(BUILD)/static.metal -o $@
$(BUILD)/paired.air: $(BUILD)/generated.stamp
	$(METAL) $(METALFLAGS) -c $(BUILD)/paired.metal -o $@
$(BUILD)/expanded.air: $(BUILD)/generated.stamp
	$(METAL) $(METALFLAGS) -c $(BUILD)/expanded.metal -o $@
$(BUILD)/sg8.air: $(BUILD)/generated.stamp
	$(METAL) $(METALFLAGS) -c $(BUILD)/sg8.metal -o $@
$(BUILD)/splash.metallib: $(BUILD)/static.air $(BUILD)/paired.air $(BUILD)/expanded.air $(BUILD)/sg8.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
cpu-self-test: $(BUILD)/oracle
	$(BUILD)/oracle --cpu-self-test
