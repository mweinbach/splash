BUILD ?= build/compact-r4-preflight-bundle-sep22-worker-v1
CXX := xcrun -sdk macosx clang++
DIR := dev/benchmarks/expert_r4_preflight_bundle_sep22
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention
include $(BUILD)/link-inputs.mk
HOST := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(REBUILD_NAMES)))
define HOST_RULE
$(BUILD)/host/$(1).o: $(SRC_$(1)) $(BUILD)/source/$(DIR)/policy.hpp $(BUILD)/source/$(DIR)/guard.hpp $(BUILD)/source/$(DIR)/source_identity.hpp $(BUILD)/overlay-manifest.json
	mkdir -p $$(dir $$@)
	$(CXX) $(FLAGS) -MMD -MP -c $$< -o $$@
endef
$(foreach name,$(REBUILD_NAMES),$(eval $(call HOST_RULE,$(name))))
.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash $(BUILD)/guard-decision-cpu $(BUILD)/preflight-policy-cpu
$(BUILD)/splash-flash: $(HOST) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/guard-decision-cpu: $(BUILD)/source/$(DIR)/guard_decision_cpu.cpp $(BUILD)/source/$(DIR)/guard.hpp $(BUILD)/source/$(DIR)/baseline_generated.hpp
	xcrun -sdk macosx clang++ -std=c++20 -O3 -Wall -Wextra -Werror -pedantic -I$(BUILD)/source/$(DIR) $< -o $@
$(BUILD)/preflight-policy-cpu: $(BUILD)/source/$(DIR)/policy_cpu.cpp $(BUILD)/source/$(DIR)/policy.hpp $(BUILD)/source/$(DIR)/source_identity.hpp
	xcrun -sdk macosx clang++ -std=c++20 -O3 -Wall -Wextra -Werror -pedantic -I$(BUILD)/source/$(DIR) $< -o $@
cpu-self-test: all
	$(BUILD)/guard-decision-cpu
	$(BUILD)/preflight-policy-cpu
	$(BUILD)/preflight-policy-cpu --missing
	$(BUILD)/preflight-policy-cpu --freeze0
	$(BUILD)/preflight-policy-cpu --freeze1
	$(BUILD)/preflight-policy-cpu --retry0
	$(BUILD)/preflight-policy-cpu --retry1
	$(BUILD)/splash-flash --cpu-self-test
-include $(HOST:.o=.d)
