BUILD ?= build/gemv-r1-ab-qsa-sep21-worker-v1
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention
include $(BUILD)/link-inputs.mk
include $(BUILD)/own-inputs.mk
OWN := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(OWN_NAMES)))
.PHONY: all cpu
all: $(BUILD)/splash-flash $(BUILD)/splash.metallib $(BUILD)/policy-cpu
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.cpp $(BUILD)/overlay-manifest.json
	@mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.mm $(BUILD)/overlay-manifest.json
	@mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/splash-flash: $(OWN) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/vector-r1.air: $(BUILD)/source/dev/benchmarks/gemv_decode_r1_worker_sep21/candidate.metal $(BUILD)/overlay-manifest.json
	xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -I$(BUILD)/source -I$(BUILD)/source/runtime -c $< -o $@
$(BUILD)/splash.metallib: $(BUILD)/vector-r1.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
$(BUILD)/policy-cpu: $(BUILD)/source/dev/benchmarks/gemv_decode_r1_worker_sep21/policy_cpu.cpp
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
