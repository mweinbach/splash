BUILD ?= build/compact-native-r4-verify-teacher-sep22-worker-v1b
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
DIR := dev/benchmarks/expert_r4_compact_verify_worker_sep22
include $(BUILD)/link-inputs.mk
HOST := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(REBUILD_NAMES)))
define HOST_RULE
$(BUILD)/host/$(1).o: $(SRC_$(1)) $(BUILD)/source/$(DIR)/bridge.hpp $(BUILD)/source/$(DIR)/source_identity.hpp $(BUILD)/overlay-manifest.json
	mkdir -p $$(dir $$@)
	$(CXX) $(FLAGS) -MMD -MP -c $$< -o $$@
endef
$(foreach name,$(REBUILD_NAMES),$(eval $(call HOST_RULE,$(name))))
.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash $(BUILD)/splash.metallib $(BUILD)/policy-cpu
$(BUILD)/splash-flash: $(HOST) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/policy-cpu: $(BUILD)/source/$(DIR)/policy_cpu.cpp $(BUILD)/source/$(DIR)/bridge.hpp $(BUILD)/source/$(DIR)/source_identity.hpp
	$(CXX) $(FLAGS) $< -o $@
$(BUILD)/compact-plan.air: $(BUILD)/source/$(DIR)/plan.metal $(BUILD)/source/$(DIR)/abi.hpp
	xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(BUILD)/source/runtime -I$(BUILD)/source/$(DIR) -mmacosx-version-min=27.0 -c $< -o $@
$(BUILD)/splash.metallib: $(BUILD)/compact-plan.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
cpu-self-test: all
	$(BUILD)/policy-cpu
	$(BUILD)/policy-cpu --freeze0
	$(BUILD)/policy-cpu --freeze1
	$(BUILD)/policy-cpu --missing
	$(BUILD)/policy-cpu --retry0
	$(BUILD)/policy-cpu --retry1
	$(BUILD)/splash-flash --cpu-self-test
-include $(HOST:.o=.d)
