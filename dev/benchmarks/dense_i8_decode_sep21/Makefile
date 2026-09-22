BUILD ?= build/dense-i8-decode-sep21
HOST ?= build/prefill4k-wide-fullcache
BASE ?= build/flash-next
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -I$(HOST)/source/runtime -Iruntime -I. -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -I. -mmacosx-version-min=27.0
FLASH := $(filter-out $(HOST)/host/FlashWorker.o,$(wildcard $(HOST)/host/*.o))
CORE := $(BASE)/engine/metal/MetalBackend.o $(BASE)/engine/metal/DeviceCapabilities.o $(BASE)/engine/engine/Protocol.o $(BASE)/engine/engine/MemoryGovernor.o
AIRS := $(wildcard $(BASE)/metal/*/*.air)
SOURCE := dev/benchmarks/dense_i8_decode_sep21

.PHONY: all cpu-self-test
all: $(BUILD)/oracle $(BUILD)/splash.metallib

$(BUILD)/oracle.o: $(SOURCE)/oracle.mm $(SOURCE)/cache.hpp $(SOURCE)/precision.hpp $(SOURCE)/abi.hpp
	mkdir -p $(BUILD)
	$(CXX) $(FLAGS) -c $< -o $@

$(BUILD)/oracle: $(BUILD)/oracle.o $(FLASH) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@

$(BUILD)/candidate.air: $(SOURCE)/candidate.metal $(SOURCE)/abi.hpp runtime/metal/kernels/common/flash_affine_mpp_common.h
	mkdir -p $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@

$(BUILD)/splash.metallib: $(BUILD)/candidate.air $(AIRS)
	mkdir -p $(BUILD)
	xcrun -sdk macosx metallib $^ -o $@

cpu-self-test: all
	$(BUILD)/oracle --cpu-self-test
