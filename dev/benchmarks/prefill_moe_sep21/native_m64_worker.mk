BUILD ?= build/prefill-moe-sep21-native-m64-worker
BASE ?= build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1
COREBASE ?= build/flash-next
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
REUSED := $(filter-out $(BASE)/host/FlashMoEBlocked.o,$(wildcard $(BASE)/host/*.o))
NONWORKER := $(filter-out $(BASE)/host/FlashWorker.o,$(REUSED))
CORE := $(COREBASE)/engine/metal/MetalBackend.o $(COREBASE)/engine/metal/DeviceCapabilities.o $(COREBASE)/engine/engine/Protocol.o $(COREBASE)/engine/engine/MemoryGovernor.o
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash $(BUILD)/prefill4k-attribution $(BUILD)/policy-cpu
$(BUILD)/FlashMoEBlocked.o: $(BUILD)/source/runtime/flash/FlashMoEBlocked.cpp $(BUILD)/overlay-manifest.json
	$(CXX) $(FLAGS) -c $< -o $@
$(BUILD)/splash-flash: $(BUILD)/FlashMoEBlocked.o $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
	cp $(BASE)/splash-flash.config $@.config
$(BUILD)/prefill4k-attribution: dev/benchmarks/prefill4k_attribution.mm $(BUILD)/FlashMoEBlocked.o $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/policy-cpu: dev/benchmarks/prefill_moe_sep21/native_m64_policy_cpu.cpp $(BUILD)/FlashMoEBlocked.o $(NONWORKER) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
cpu-self-test: $(BUILD)/policy-cpu $(BUILD)/splash-flash
	$(BUILD)/policy-cpu
	$(BUILD)/splash-flash --cpu-self-test
