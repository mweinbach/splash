BUILD ?= build/hc-pad-compact-r4-verify-teacher-sep22-worker-v1
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
include $(BUILD)/link-inputs.mk
HOST := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(REBUILD_NAMES)))
define HOST_RULE
$(BUILD)/host/$(1).o: $(SRC_$(1)) $(BUILD)/source/dev/benchmarks/hc_pad_verify_worker_sep22/worker_bridge.hpp $(BUILD)/overlay-manifest.json
	mkdir -p $$(dir $$@)
	$(CXX) $(FLAGS) -MMD -MP -c $$< -o $$@
endef
$(foreach name,$(REBUILD_NAMES),$(eval $(call HOST_RULE,$(name))))
.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash $(BUILD)/splash.metallib $(BUILD)/hc-policy-cpu $(BUILD)/compact-policy-cpu
$(BUILD)/splash-flash: $(HOST) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/hc-policy-cpu: $(BUILD)/source/dev/benchmarks/hc_pad_verify_worker_sep22/policy_cpu.cpp
	$(CXX) $(FLAGS) $< -o $@
$(BUILD)/compact-policy-cpu: $(BUILD)/source/dev/benchmarks/expert_r4_compact_verify_worker_sep22/policy_cpu.cpp
	$(CXX) $(FLAGS) $< -o $@
$(BUILD)/hc-pad.air: $(BUILD)/source/dev/benchmarks/hc_pad_producer_sep22/candidate.metal $(BUILD)/source/dev/benchmarks/hc_pad_producer_sep22/abi.hpp
	xcrun -sdk macosx metal -std=metal4.1 -O3 -mmacosx-version-min=27.0 -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/hc_pad_producer_sep22 -c $< -o $@
$(BUILD)/splash.metallib: $(BUILD)/hc-pad.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
cpu-self-test: all
	$(BUILD)/hc-policy-cpu
	$(BUILD)/compact-policy-cpu
	$(BUILD)/compact-policy-cpu --freeze0
	$(BUILD)/compact-policy-cpu --freeze1
	$(BUILD)/compact-policy-cpu --missing
	$(BUILD)/compact-policy-cpu --retry0
	$(BUILD)/compact-policy-cpu --retry1
	$(BUILD)/splash-flash --cpu-self-test
-include $(HOST:.o=.d)
