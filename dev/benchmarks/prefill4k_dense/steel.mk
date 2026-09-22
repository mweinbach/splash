# Compile/link only. This Makefile never submits GPU work.
STEEL_BUILD ?= build/prefill4k-steel
STEEL_SOURCE := dev/benchmarks/prefill4k_dense/steel_adapter.metal
STEEL_VENDOR := dev/benchmarks/prefill4k_dense/steel_vendor
STEEL_HEADERS := $(shell rg --files $(STEEL_VENDOR) -g '*.h')
STEEL_METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -I$(STEEL_VENDOR) -mmacosx-version-min=27.0

.PHONY: steel-all
steel-all: $(STEEL_BUILD)/steel.metallib

$(STEEL_BUILD):
	mkdir -p $@

$(STEEL_BUILD)/steel_adapter.air: $(STEEL_SOURCE) $(STEEL_HEADERS) runtime/metal/abi/FlashDenseCache.h \
    runtime/metal/kernels/common/flash_dense_traversal.h | $(STEEL_BUILD)
	xcrun -sdk macosx metal $(STEEL_METALFLAGS) -c $< -o $@

$(STEEL_BUILD)/steel.metallib: $(STEEL_BUILD)/steel_adapter.air
	xcrun -sdk macosx metallib $< -o $@
