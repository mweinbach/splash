# Isolated source build. No target executes Metal or creates a GPU resource.
GDN_CHUNK_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)
GDN_CHUNK_SOURCE := $(GDN_CHUNK_ROOT)/dev/benchmarks/gdn_chunk_sep21
GDN_CHUNK_BUILD ?= $(GDN_CHUNK_ROOT)/build/gdn-chunk-sep21
GDN_CHUNK_CXX := xcrun -sdk macosx clang++
GDN_CHUNK_FLAGS := -std=c++20 -O3 -ffp-contract=off -fno-fast-math -Wall -Wextra -Werror -Wno-deprecated-declarations -I$(GDN_CHUNK_ROOT)/runtime -mmacosx-version-min=27.0
GDN_CHUNK_METAL_FLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(GDN_CHUNK_ROOT)/runtime -mmacosx-version-min=27.0

.PHONY: all
all: $(GDN_CHUNK_BUILD)/cpu-oracle $(GDN_CHUNK_BUILD)/metal-oracle $(GDN_CHUNK_BUILD)/gdn-chunk.metallib

$(GDN_CHUNK_BUILD):
	mkdir -p $@

$(GDN_CHUNK_BUILD)/cpu-oracle.o: $(GDN_CHUNK_SOURCE)/cpu_oracle.cpp $(GDN_CHUNK_SOURCE)/cpu_oracle.hpp $(GDN_CHUNK_SOURCE)/Makefile | $(GDN_CHUNK_BUILD)
	$(GDN_CHUNK_CXX) $(GDN_CHUNK_FLAGS) -DGDN_CPU_ORACLE_NO_MAIN -c $< -o $@

$(GDN_CHUNK_BUILD)/cpu-oracle: $(GDN_CHUNK_SOURCE)/cpu_oracle.cpp $(GDN_CHUNK_SOURCE)/cpu_oracle.hpp $(GDN_CHUNK_SOURCE)/Makefile | $(GDN_CHUNK_BUILD)
	$(GDN_CHUNK_CXX) $(GDN_CHUNK_FLAGS) $< -o $@

$(GDN_CHUNK_BUILD)/metal-oracle: $(GDN_CHUNK_SOURCE)/metal_oracle.mm $(GDN_CHUNK_SOURCE)/cpu_oracle.hpp $(GDN_CHUNK_SOURCE)/Makefile $(GDN_CHUNK_ROOT)/runtime/metal/abi/FlashGDN.h $(GDN_CHUNK_BUILD)/cpu-oracle.o
	$(GDN_CHUNK_CXX) $(GDN_CHUNK_FLAGS) -fobjc-arc $< $(GDN_CHUNK_BUILD)/cpu-oracle.o -framework Foundation -framework Metal -o $@

$(GDN_CHUNK_BUILD)/candidate.air: $(GDN_CHUNK_SOURCE)/candidate.metal $(GDN_CHUNK_ROOT)/runtime/metal/abi/FlashGDN.h | $(GDN_CHUNK_BUILD)
	xcrun -sdk macosx metal $(GDN_CHUNK_METAL_FLAGS) -c $< -o $@

$(GDN_CHUNK_BUILD)/canonical.air: $(GDN_CHUNK_ROOT)/runtime/metal/kernels/shared/flash_gdn_staged.metal $(GDN_CHUNK_ROOT)/runtime/metal/abi/FlashGDN.h | $(GDN_CHUNK_BUILD)
	xcrun -sdk macosx metal $(GDN_CHUNK_METAL_FLAGS) -c $< -o $@

$(GDN_CHUNK_BUILD)/gdn-chunk.metallib: $(GDN_CHUNK_BUILD)/candidate.air $(GDN_CHUNK_BUILD)/canonical.air
	xcrun -sdk macosx metallib $^ -o $@
