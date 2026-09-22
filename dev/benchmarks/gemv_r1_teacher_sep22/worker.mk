BUILD ?= build/gemv-r1-teacher-sep22-worker-v1
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
include $(BUILD)/link-inputs.mk
OWN := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(OWN_NAMES)))
DIR := dev/benchmarks/gemv_r1_teacher_sep22
define HOST_RULE
$(BUILD)/host/$(1).o: $(SRC_$(1)) $(BUILD)/overlay-manifest.json $(BUILD)/source/$(DIR)/scope.hpp
	mkdir -p $$(dir $$@)
	$(CXX) $(FLAGS) -MMD -MP -c $$< -o $$@
endef
$(foreach name,$(OWN_NAMES),$(eval $(call HOST_RULE,$(name))))
.PHONY: all cpu
all: $(BUILD)/splash-flash $(BUILD)/splash.metallib $(BUILD)/policy-cpu
$(BUILD)/splash-flash: $(OWN) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
$(BUILD)/splash.metallib: $(AIRS) $(BUILD)/baseline-original76.metallib
	xcrun -sdk macosx metallib $(AIRS) -o $@
$(BUILD)/policy-cpu: $(BUILD)/source/$(DIR)/policy_cpu.cpp $(BUILD)/source/$(DIR)/scope.hpp
	$(CXX) $(FLAGS) $< -o $@
cpu: all
	$(BUILD)/policy-cpu
	$(BUILD)/policy-cpu --freeze0
	$(BUILD)/policy-cpu --freeze1
	$(BUILD)/policy-cpu --missing
	$(BUILD)/policy-cpu --retry0
	$(BUILD)/policy-cpu --retry1
	$(BUILD)/splash-flash --cpu-self-test
-include $(OWN:.o=.d)
