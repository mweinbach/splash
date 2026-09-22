BUILD ?= build/bf16-target-r4-pointwise-sep21-v2
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention
CHANGED := $(addprefix $(BUILD)/host/,$(addsuffix .o,FlashForward FlashInt8ExpertStore FlashWorker FlashBatchForward FlashBatchPrefill FlashBatchVerify))
include $(BUILD)/link-inputs.mk
.PHONY: all cpu
all: $(BUILD)/splash-flash $(BUILD)/splash.metallib $(BUILD)/policy-cpu
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.cpp $(BUILD)/overlay-manifest.json
	@mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.mm $(BUILD)/overlay-manifest.json
	@mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/splash-flash: $(CHANGED) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/policy-cpu: $(BUILD)/source/dev/benchmarks/decode_bf16_target_r4_sep21/policy_cpu.cpp $(BUILD)/overlay-manifest.json
	$(CXX) $(FLAGS) $< -o $@
$(BUILD)/narrow.air: $(BUILD)/source/dev/benchmarks/decode_bf16_narrow_sep21/kernels.metal $(BUILD)/overlay-manifest.json
	xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/decode_bf16_narrow_sep21 -c $< -o $@
$(BUILD)/splash.metallib: $(BUILD)/narrow.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
cpu: all
	$(BUILD)/policy-cpu
	$(BUILD)/splash-flash --cpu-self-test
-include $(CHANGED:.o=.d)
