BUILD ?= build/prefill-moe-sep21-w8a8-wide
HOST ?= build/prefill4k-wide-fullcache
BASE ?= build/flash-next
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -I$(HOST)/source/runtime -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -I. -mmacosx-version-min=27.0
FLASH := $(filter-out $(HOST)/host/FlashWorker.o,$(wildcard $(HOST)/host/*.o))
CORE := $(BASE)/engine/metal/MetalBackend.o $(BASE)/engine/metal/DeviceCapabilities.o $(BASE)/engine/engine/Protocol.o $(BASE)/engine/engine/MemoryGovernor.o
AIRS := $(filter-out $(BASE)/metal/shared/flash_int8_expert_store.air,$(wildcard $(BASE)/metal/*/*.air)) $(HOST)/metal/flash_int8_expert_store.air
ABI_HEADERS := runtime/metal/abi/FlashInt8ExpertStore.h runtime/metal/abi/FlashMoEBuckets.h
COMMON_HEADER := runtime/metal/kernels/common/flash_affine_mpp_common.h

.PHONY: all cpu-self-test
all: $(BUILD)/oracle $(BUILD)/splash.metallib

$(BUILD)/oracle.mm: dev/benchmarks/prefill_moe_sep21/w8a8_wide/generate.py dev/benchmarks/prefill_moe_sep21/w8a8_wide/cache.hpp dev/benchmarks/prefill_moe_sep21/w8a8/generate.py dev/benchmarks/prefill_moe_sep21/w8a8/cache.hpp dev/benchmarks/prefill_moe_sep21/w8a8/precision.hpp dev/benchmarks/prefill_moe_sep21/native_m64/generate.py build/prefill4k-int8tiles/oracle.mm
	mkdir -p $(BUILD)
	.venv/bin/python $< $(BUILD)

$(BUILD)/oracle.o: $(BUILD)/oracle.mm dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp $(ABI_HEADERS)
	$(CXX) $(FLAGS) -c $< -o $@

$(BUILD)/oracle: $(BUILD)/oracle.o $(FLASH) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@

$(BUILD)/baseline.metal: dev/benchmarks/prefill_moe_sep21/w8a8/generate_kernel.py dev/benchmarks/prefill_moe_sep21/memory.metal dev/benchmarks/prefill_moe_sep21/w8a8/candidate.metal
	mkdir -p $(BUILD)
	.venv/bin/python $< $@
	cmp $@ dev/benchmarks/prefill_moe_sep21/w8a8/candidate.metal

$(BUILD)/baseline.air: $(BUILD)/baseline.metal $(ABI_HEADERS) $(COMMON_HEADER)
	$(METAL) $(METALFLAGS) -c $< -o $@

$(BUILD)/candidate.metal: dev/benchmarks/prefill_moe_sep21/w8a8_wide/generate_kernel.py dev/benchmarks/prefill_moe_sep21/w8a8/candidate.metal
	mkdir -p $(BUILD)
	.venv/bin/python $< $@

$(BUILD)/candidate.air: $(BUILD)/candidate.metal $(ABI_HEADERS) $(COMMON_HEADER)
	$(METAL) $(METALFLAGS) -c $< -o $@

$(BUILD)/splash.metallib: $(BUILD)/baseline.air $(BUILD)/candidate.air $(AIRS)
	mkdir -p $(BUILD)
	xcrun -sdk macosx metallib $^ -o $@

cpu-self-test: all
	$(BUILD)/oracle --cpu-self-test
