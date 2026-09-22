BUILD ?= build/adaptive-expert-tail-sg2k128-sep21
HOST ?= build/prefill4k-wide-fullcache
BASE ?= build/flash-next
FIXED_SOURCE ?= build/prefill-moe-sg2-k128-pointwise-sep21-worker-v1/source/dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker/candidate.metal
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
PYTHON ?= .venv/bin/python
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -I$(HOST)/source/runtime -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
FLASH := $(filter-out $(HOST)/host/FlashWorker.o,$(wildcard $(HOST)/host/*.o))
CORE := $(BASE)/engine/metal/MetalBackend.o $(BASE)/engine/metal/DeviceCapabilities.o $(BASE)/engine/engine/Protocol.o $(BASE)/engine/engine/MemoryGovernor.o
AIRS := $(filter-out $(BASE)/metal/shared/flash_int8_expert_store.air,$(wildcard $(BASE)/metal/*/*.air)) $(HOST)/metal/flash_int8_expert_store.air
DIR := dev/benchmarks/adaptive_expert_tail_sep21/combined

.PHONY: all cpu-self-test source-check
all: $(BUILD)/oracle $(BUILD)/splash.metallib

$(BUILD)/oracle.mm: $(DIR)/generate_oracle.py dev/benchmarks/adaptive_expert_tail_sep21/generate_oracle.py dev/benchmarks/prefill_moe_sep21/native_m16/generate.py dev/benchmarks/prefill_moe_sep21/native_m64/generate.py build/prefill4k-int8tiles/oracle.mm
	mkdir -p $(BUILD)
	$(PYTHON) $< $(BUILD)

$(BUILD)/oracle.o: $(BUILD)/oracle.mm dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp
	$(CXX) $(FLAGS) -c $< -o $@

$(BUILD)/oracle: $(BUILD)/oracle.o $(FLASH) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@

$(BUILD)/adaptive.metal: $(DIR)/generate_shader.py $(FIXED_SOURCE)
	mkdir -p $(BUILD)
	$(PYTHON) $< $(BUILD)

$(BUILD)/adaptive.air: $(BUILD)/adaptive.metal
	$(METAL) $(METALFLAGS) -c $< -o $@

# Compile the exact frozen variant7 source separately. Its non-probe exports
# remain the baseline to which the private M32 probe is strictly qualified.
$(BUILD)/variant7.air: $(FIXED_SOURCE)
	mkdir -p $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@

$(BUILD)/splash.metallib: $(BUILD)/adaptive.air $(BUILD)/variant7.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@

source-check: $(BUILD)/adaptive.metal $(BUILD)/oracle.mm
	$(PYTHON) $(DIR)/run.py --build $(BUILD) --source-check

cpu-self-test: all source-check
	$(BUILD)/oracle --cpu-self-test
