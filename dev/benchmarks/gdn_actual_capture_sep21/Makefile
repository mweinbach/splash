ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)
SOURCE := $(ROOT)/dev/benchmarks/gdn_actual_capture_sep21
V6 := $(ROOT)/dev/benchmarks/gdn_nax_chunks_sep21_v6
BUILD := $(ROOT)/build/gdn-actual-capture-sep21-v1
CXX := xcrun -sdk macosx clang++
CXXFLAGS := -std=c++20 -O3 -ffp-contract=off -fno-fast-math -Wall -Wextra -Werror -Wno-deprecated-declarations -I$(V6)/frozen -mmacosx-version-min=27.0
.PHONY: all
all: $(BUILD)/metal-actual-capture
$(BUILD):
	mkdir -p $@
$(BUILD)/cpu-oracle.o: $(V6)/frozen/cpu_oracle.cpp $(V6)/frozen/cpu_oracle.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -DGDN_CPU_ORACLE_NO_MAIN -c $< -o $@
$(BUILD)/metal-actual-capture: $(SOURCE)/metal_actual_capture.mm $(V6)/metal_oracle.mm $(V6)/wy_cpu.hpp $(BUILD)/cpu-oracle.o
	$(CXX) $(CXXFLAGS) -fobjc-arc $< $(BUILD)/cpu-oracle.o -framework Foundation -framework Metal -o $@
