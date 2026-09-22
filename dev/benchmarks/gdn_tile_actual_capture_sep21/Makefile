ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)
SOURCE := $(ROOT)/dev/benchmarks/gdn_tile_actual_capture_sep21
COMPONENT := $(ROOT)/dev/benchmarks/gdn_tile_chunk_sep21
BUILD := $(ROOT)/build/gdn-tile-actual-capture-sep21-v1
CXX := xcrun -sdk macosx clang++
CXXFLAGS := -std=c++20 -O3 -ffp-contract=off -fno-fast-math -Wall -Wextra -Werror -Wno-deprecated-declarations -I$(COMPONENT) -I$(COMPONENT)/frozen -mmacosx-version-min=27.0
.PHONY: all
all: $(BUILD)/metal-actual-tile
$(BUILD):
	mkdir -p $@
$(BUILD)/cpu-oracle.o: $(COMPONENT)/frozen/cpu_oracle.cpp $(COMPONENT)/frozen/cpu_oracle.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DGDN_CPU_ORACLE_NO_MAIN -c $< -o $@
$(BUILD)/metal-actual-tile: $(SOURCE)/metal_actual_capture.mm $(COMPONENT)/metal_oracle.mm $(COMPONENT)/abi.hpp $(BUILD)/cpu-oracle.o
	$(CXX) $(CXXFLAGS) -fobjc-arc $< $(BUILD)/cpu-oracle.o -framework Foundation -framework Metal -o $@
