# Isolated compile-only default. No target creates a Metal device or GPU resource.
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)
SOURCE := $(ROOT)/dev/benchmarks/gdn_nax_chunks_sep21
BUILD ?= $(ROOT)/build/gdn-nax-chunks-sep21
CXX := xcrun -sdk macosx clang++
CXXFLAGS := -std=c++20 -O3 -ffp-contract=off -fno-fast-math -Wall -Wextra -Werror -Wno-deprecated-declarations -I$(SOURCE)/frozen -mmacosx-version-min=27.0
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(SOURCE)/frozen -mmacosx-version-min=27.0
.PHONY: all
all: $(BUILD)/metal-oracle-wy $(BUILD)/gdn-wy.metallib $(BUILD)/cpu-wy-reference
$(BUILD):
	mkdir -p $@
$(BUILD)/cpu-oracle.o: $(SOURCE)/frozen/cpu_oracle.cpp $(SOURCE)/frozen/cpu_oracle.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DGDN_CPU_ORACLE_NO_MAIN -c $< -o $@
$(BUILD)/metal-oracle-wy: $(SOURCE)/metal_oracle.mm $(SOURCE)/wy_cpu.hpp $(BUILD)/cpu-oracle.o
	$(CXX) $(CXXFLAGS) -fobjc-arc $< $(BUILD)/cpu-oracle.o -framework Foundation -framework Metal -o $@
$(BUILD)/candidate.air: $(SOURCE)/candidate.metal $(SOURCE)/frozen/metal/abi/FlashGDN.h | $(BUILD)
	xcrun -sdk macosx metal $(METALFLAGS) -c $< -o $@
$(BUILD)/canonical.air: $(SOURCE)/frozen/canonical.metal $(SOURCE)/frozen/metal/abi/FlashGDN.h | $(BUILD)
	xcrun -sdk macosx metal $(METALFLAGS) -c $< -o $@
$(BUILD)/gdn-wy.metallib: $(BUILD)/candidate.air $(BUILD)/canonical.air
	xcrun -sdk macosx metallib $^ -o $@

$(BUILD)/cpu-wy-reference: $(SOURCE)/wy_cpu_test.cpp $(SOURCE)/wy_cpu.hpp $(BUILD)/cpu-oracle.o
	$(CXX) $(CXXFLAGS) $< $(BUILD)/cpu-oracle.o -o $@
