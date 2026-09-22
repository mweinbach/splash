# CPU-only source build. Root explicitly invokes --gpu after the exact gate.
COMPACT_BUILD ?= build/expert-r4-compact-native-sep22-component-v2
include $(COMPACT_BUILD)/component-inputs.mk
COMPACT_DIR := dev/benchmarks/expert_r4_compact_native_parallel_sep22
COMPACT_SOURCE ?= $(COMPACT_BUILD)/source
COMPACT_HOST := $(COMPACT_FROZEN_HOST) $(COMPACT_BUILD)/host/MetalBackend.o
COMPACT_AIRS := $(COMPACT_FROZEN_AIRS)
COMPACT_FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -ffp-contract=off -fno-fast-math -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc -I$(COMPACT_SOURCE) -I$(COMPACT_SOURCE)/runtime -I$(COMPACT_SOURCE)/$(COMPACT_DIR) -I$(COMPACT_SOURCE)/dev/benchmarks -I$(COMPACT_SOURCE)/dev/benchmarks/prefill4k_attention
$(COMPACT_BUILD)/host/MetalBackend.o: $(COMPACT_SOURCE)/runtime/metal/MetalBackend.mm $(COMPACT_SOURCE)/runtime/metal/MetalBackend.hpp
	@mkdir -p $(dir $@)
	xcrun -sdk macosx clang++ $(COMPACT_FLAGS) -c $< -o $@
$(COMPACT_BUILD)/oracle: $(COMPACT_SOURCE)/$(COMPACT_DIR)/oracle.mm $(COMPACT_SOURCE)/$(COMPACT_DIR)/abi.hpp $(COMPACT_SOURCE)/$(COMPACT_DIR)/metadata.hpp $(COMPACT_SOURCE)/$(COMPACT_DIR)/quality.hpp $(COMPACT_SOURCE)/$(COMPACT_DIR)/safety.hpp $(COMPACT_HOST)
	@mkdir -p $(dir $@)
	xcrun -sdk macosx clang++ $(COMPACT_FLAGS) $< $(COMPACT_HOST) -framework Foundation -framework Metal -framework IOKit -o $@
$(COMPACT_BUILD)/compact-probe.air: $(COMPACT_SOURCE)/$(COMPACT_DIR)/probe.metal $(COMPACT_SOURCE)/$(COMPACT_DIR)/plan.metal $(COMPACT_SOURCE)/$(COMPACT_DIR)/abi.hpp
	@mkdir -p $(dir $@)
	xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(COMPACT_SOURCE) -I$(COMPACT_SOURCE)/runtime -I$(COMPACT_SOURCE)/$(COMPACT_DIR) -I$(COMPACT_SOURCE)/dev/benchmarks -mmacosx-version-min=27.0 -c $< -o $@
$(COMPACT_BUILD)/splash.metallib: $(COMPACT_BUILD)/compact-probe.air $(COMPACT_AIRS)
	xcrun -sdk macosx metallib $^ -o $@
.PHONY: compact-native-r4-cpu
compact-native-r4-cpu: $(COMPACT_BUILD)/oracle $(COMPACT_BUILD)/splash.metallib
	$(COMPACT_BUILD)/oracle --cpu-self-test
