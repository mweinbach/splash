BUILD ?= build/prefill-i8-lut-sep21-v2
HOST ?= build/prefill4k-wide-fullcache
BASE ?= build/flash-next
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -I$(HOST)/source/runtime -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -I. -mmacosx-version-min=27.0
FLASH := $(filter-out $(HOST)/host/FlashWorker.o,$(wildcard $(HOST)/host/*.o))
CORE := $(BASE)/engine/metal/MetalBackend.o $(BASE)/engine/metal/DeviceCapabilities.o $(BASE)/engine/engine/Protocol.o $(BASE)/engine/engine/MemoryGovernor.o
AIRS := $(filter-out $(BASE)/metal/shared/flash_int8_expert_store.air,$(wildcard $(BASE)/metal/*/*.air)) $(HOST)/metal/flash_int8_expert_store.air

.PHONY: all host cpu-self-test
all: $(BUILD)/oracle $(BUILD)/splash.metallib
host: $(BUILD)/oracle

$(BUILD)/oracle.mm: dev/benchmarks/prefill_i8_lut_sep21/v2/generate.py dev/benchmarks/prefill_i8_lut_sep21/v2/cache.hpp dev/benchmarks/prefill_i8_lut_sep21/v2/guard_reference.hpp dev/benchmarks/prefill_moe_sep21/native_m64/generate.py build/prefill4k-int8tiles/oracle.mm
	mkdir -p $(BUILD)
	.venv/bin/python $< $(BUILD)

$(BUILD)/oracle.o: $(BUILD)/oracle.mm dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp
	$(CXX) $(FLAGS) -c $< -o $@

$(BUILD)/oracle: $(BUILD)/oracle.o $(FLASH) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@

# The v2 oracle reuses the v1 Metal library byte-for-byte.
$(BUILD)/candidate.metal: build/prefill-i8-lut-sep21/candidate.metal
	mkdir -p $(BUILD)
	cp $< $@

$(BUILD)/candidate.air: $(BUILD)/candidate.metal
	$(METAL) $(METALFLAGS) -c $< -o $@

$(BUILD)/splash.metallib: build/prefill-i8-lut-sep21/splash.metallib
	mkdir -p $(BUILD)
	cp $< $@

cpu-self-test: host
	$(BUILD)/oracle --cpu-self-test
