BUILD ?= build/guard-HC-fast-composite-sep22-worker-v1
CXX := xcrun -sdk macosx clang++
DIR := dev/benchmarks/guard_hc_fast_composite_sep22
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention
include $(BUILD)/link-inputs.mk
HOST := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(REBUILD_NAMES)))
define HOST_RULE
$(BUILD)/host/$(1).o: $(SRC_$(1)) $(BUILD)/source/$(DIR)/policy.hpp $(BUILD)/source/$(DIR)/source_identity.hpp $(BUILD)/overlay-manifest.json
	mkdir -p $$(dir $$@)
	$(CXX) $(FLAGS) -MMD -MP -c $$< -o $$@
endef
$(foreach name,$(REBUILD_NAMES),$(eval $(call HOST_RULE,$(name))))
.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash $(BUILD)/splash.metallib
$(BUILD)/splash-flash: $(HOST) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/splash.metallib: $(AIRS) $(BUILD)/hc-pad.air
	xcrun -sdk macosx metallib $^ -o $@
cpu-self-test: all
	$(BUILD)/splash-flash --cpu-self-test
-include $(HOST:.o=.d)
