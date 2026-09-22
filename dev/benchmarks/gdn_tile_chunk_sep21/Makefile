# Compile-only default. Root alone creates a GPU device or runs GPU modes.
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)
SOURCE := $(ROOT)/dev/benchmarks/gdn_tile_chunk_sep21
BUILD ?= $(ROOT)/build/gdn-tile-chunk-sep21
CXX := xcrun -sdk macosx clang++
CXXFLAGS := -std=c++20 -O3 -ffp-contract=off -fno-fast-math -Wall -Wextra -Werror -Wno-deprecated-declarations -I$(SOURCE) -I$(SOURCE)/frozen -mmacosx-version-min=27.0
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(SOURCE) -I$(SOURCE)/frozen -mmacosx-version-min=27.0
.PHONY: all
all: $(BUILD)/metal-oracle-tile $(BUILD)/gdn-tile.metallib $(BUILD)/cpu-wy-reference
$(BUILD):
	mkdir -p $@
$(BUILD)/cpu-oracle.o: $(SOURCE)/frozen/cpu_oracle.cpp $(SOURCE)/frozen/cpu_oracle.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DGDN_CPU_ORACLE_NO_MAIN -c $< -o $@
$(BUILD)/cpu-wy-reference: $(SOURCE)/wy_cpu_test.cpp $(SOURCE)/wy_cpu.hpp $(BUILD)/cpu-oracle.o
	$(CXX) $(CXXFLAGS) $< $(BUILD)/cpu-oracle.o -o $@
$(BUILD)/metal-oracle-tile: $(SOURCE)/metal_oracle.mm $(SOURCE)/abi.hpp $(SOURCE)/wy_cpu.hpp $(BUILD)/cpu-oracle.o
	$(CXX) $(CXXFLAGS) -fobjc-arc $< $(BUILD)/cpu-oracle.o -framework Foundation -framework Metal -o $@
$(BUILD)/candidate.air: $(SOURCE)/candidate.metal $(SOURCE)/native_helper.metal $(SOURCE)/native_audit_helper.metal $(SOURCE)/abi.hpp $(SOURCE)/frozen/metal/abi/FlashGDN.h | $(BUILD)
	xcrun -sdk macosx metal $(METALFLAGS) -c $< -o $@
$(BUILD)/canonical.air: $(SOURCE)/frozen/canonical.metal $(SOURCE)/frozen/metal/abi/FlashGDN.h | $(BUILD)
	xcrun -sdk macosx metal $(METALFLAGS) -c $< -o $@
$(BUILD)/v6-math-control.air: $(SOURCE)/frozen/v6_math_control.metal $(SOURCE)/abi.hpp $(SOURCE)/frozen/metal/abi/FlashGDN.h | $(BUILD)
	xcrun -sdk macosx metal $(METALFLAGS) -c $< -o $@
$(BUILD)/gdn-tile.metallib: $(BUILD)/candidate.air $(BUILD)/canonical.air $(BUILD)/v6-math-control.air
	xcrun -sdk macosx metallib $^ -o $@
