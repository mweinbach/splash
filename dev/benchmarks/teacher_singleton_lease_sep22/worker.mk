BUILD ?= build/teacher-singleton-lease-sep22-worker-v1
.DEFAULT_GOAL := all
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention
include $(BUILD)/link-inputs.mk
HOST := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(REBUILD_NAMES)))
define HOST_RULE
$(BUILD)/host/$(1).o: $(SRC_$(1)) $(BUILD)/overlay-manifest.json
	mkdir -p $$(dir $$@)
	$(CXX) $(FLAGS) -MMD -MP -c $$< -o $$@
endef
$(foreach name,$(REBUILD_NAMES),$(eval $(call HOST_RULE,$(name))))
.PHONY: all cpu
all: $(BUILD)/splash-flash
$(BUILD)/splash-flash: $(HOST) $(CORE)
	$(CXX) $(FLAGS) $(HOST) $(CORE) -framework Foundation -framework Metal -framework IOKit -o $@
cpu: all
	$(BUILD)/splash-flash --cpu-self-test
-include $(HOST:.o=.d)
