BUILD ?= build/gdn-ab-merge-hc-v3-sep21-worker-v1
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention
include $(BUILD)/link-inputs.mk
.PHONY: all cpu
all: $(BUILD)/splash-flash $(BUILD)/splash.metallib
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.cpp $(BUILD)/overlay-manifest.json
	@mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/%.o: $(BUILD)/source/runtime/flash/%.mm $(BUILD)/overlay-manifest.json
	@mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/splash-flash: $(BUILD)/host/FlashForward.o $(BUILD)/host/FlashWorker.o $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/candidate.air: $(BUILD)/source/dev/benchmarks/gdn_ab_merge_sep21/candidate.metal $(BUILD)/overlay-manifest.json
	xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/gdn_ab_merge_sep21 -c $< -o $@
$(BUILD)/splash.metallib: $(BUILD)/candidate.air $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
cpu: all
	$(BUILD)/splash-flash --cpu-self-test
-include $(BUILD)/host/FlashForward.d $(BUILD)/host/FlashWorker.d
