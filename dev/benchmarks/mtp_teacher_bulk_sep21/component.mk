BUILD ?= build/mtp-teacher-bulk-sep21-v1
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -fobjc-arc -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
DIR := dev/benchmarks/mtp_teacher_bulk_sep21
include $(BUILD)/link-inputs.mk
HOST := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(REBUILD_NAMES)))
define HOST_RULE
$(BUILD)/host/$(1).o: $(SRC_$(1)) $(BUILD)/source/runtime/flash/FlashMTP.hpp $(BUILD)/source-manifest.json
	mkdir -p $$(dir $$@)
	$(CXX) $(FLAGS) -MMD -MP -c $$< -o $$@
endef
$(foreach name,$(REBUILD_NAMES),$(eval $(call HOST_RULE,$(name))))
.PHONY: all cpu-self-test
all: $(BUILD)/teacher-bulk-oracle $(BUILD)/policy-cpu
$(BUILD)/host/teacher_bulk.o: $(BUILD)/source/$(DIR)/bulk.cpp $(BUILD)/source/$(DIR)/bulk.hpp $(BUILD)/source/runtime/flash/FlashMTP.hpp
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/teacher-bulk-oracle: $(BUILD)/source/$(DIR)/oracle.mm $(BUILD)/host/teacher_bulk.o $(HOST) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) -MMD -MP $(BUILD)/source/$(DIR)/oracle.mm $(BUILD)/host/teacher_bulk.o $(HOST) $(REUSED) $(CORE) $(FRAMEWORKS) -o $@
$(BUILD)/policy-cpu: $(BUILD)/source/$(DIR)/policy_cpu.cpp $(BUILD)/source/$(DIR)/bulk.hpp
	$(CXX) $(FLAGS) -MMD -MP $< -o $@
cpu-self-test: all
	$(BUILD)/policy-cpu
	$(BUILD)/teacher-bulk-oracle --help
-include $(HOST:.o=.d) $(BUILD)/host/teacher_bulk.d $(BUILD)/teacher-bulk-oracle.d $(BUILD)/policy-cpu.d
