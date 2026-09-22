# CPU-build only. Root explicitly invokes GPU mode; this target opens no model.
COHORT_BUILD ?= build/expert-r4-cohort-n16-sep22-component-v1
include $(COHORT_BUILD)/component-inputs.mk
COHORT_DIR := dev/benchmarks/expert_r4_cohort_sep22
COHORT_SOURCE ?= $(COHORT_BUILD)/source
COHORT_HOST := $(COHORT_FROZEN_HOST) $(COHORT_BUILD)/host/MetalBackend.o
# probe.metal/control-tap linkage is supplied by the isolated source generator.
# It must preserve currentSG4 arithmetic and export each existing kernel once.
COHORT_AIRS := $(COHORT_FROZEN_AIRS)
COHORT_FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -ffp-contract=off -fno-fast-math -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc -I$(COHORT_SOURCE) -I$(COHORT_SOURCE)/runtime -I$(COHORT_SOURCE)/$(COHORT_DIR) -I$(COHORT_SOURCE)/dev/benchmarks -I$(COHORT_SOURCE)/dev/benchmarks/prefill4k_attention
$(COHORT_BUILD)/host/MetalBackend.o: $(COHORT_SOURCE)/runtime/metal/MetalBackend.mm $(COHORT_SOURCE)/runtime/metal/MetalBackend.hpp
	@mkdir -p $(dir $@)
	xcrun -sdk macosx clang++ $(COHORT_FLAGS) -c $< -o $@
$(COHORT_BUILD)/oracle: $(COHORT_SOURCE)/$(COHORT_DIR)/oracle.mm $(COHORT_SOURCE)/$(COHORT_DIR)/quality.hpp $(COHORT_SOURCE)/$(COHORT_DIR)/abi.hpp $(COHORT_SOURCE)/$(COHORT_DIR)/malformed.hpp $(COHORT_HOST)
	@mkdir -p $(dir $@)
	xcrun -sdk macosx clang++ $(COHORT_FLAGS) $< $(COHORT_HOST) -framework Foundation -framework Metal -framework IOKit -o $@
$(COHORT_BUILD)/cohort-probe.air: $(COHORT_SOURCE)/$(COHORT_DIR)/probe.metal $(COHORT_SOURCE)/$(COHORT_DIR)/kernels.metal $(COHORT_SOURCE)/$(COHORT_DIR)/abi.hpp
	@mkdir -p $(dir $@)
	xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(COHORT_SOURCE) -I$(COHORT_SOURCE)/runtime -I$(COHORT_SOURCE)/$(COHORT_DIR) -I$(COHORT_SOURCE)/dev/benchmarks -mmacosx-version-min=27.0 -c $< -o $@
$(COHORT_BUILD)/splash.metallib: $(COHORT_BUILD)/cohort-probe.air $(COHORT_AIRS)
	xcrun -sdk macosx metallib $^ -o $@
.PHONY: expert-r4-cohort-cpu expert-r4-cohort-host-cpu
expert-r4-cohort-cpu: $(COHORT_BUILD)/oracle $(COHORT_BUILD)/splash.metallib
	$(COHORT_BUILD)/oracle --cpu-self-test
expert-r4-cohort-host-cpu: $(COHORT_BUILD)/oracle
	$(COHORT_BUILD)/oracle --cpu-self-test
