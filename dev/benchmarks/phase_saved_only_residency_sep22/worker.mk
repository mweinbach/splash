BUILD ?= build/phase-saved-only-residency-sep22-worker-v1
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention
include $(BUILD)/link-inputs.mk
.PHONY: all cpu
all: $(BUILD)/splash-flash
$(BUILD)/host/FlashWorker.o: $(BUILD)/source/runtime/flash/FlashWorker.mm $(BUILD)/overlay-manifest.json
	mkdir -p $(dir $@)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/splash-flash: $(BUILD)/host/FlashWorker.o $(REUSED)
	$(CXX) $(FLAGS) $(BUILD)/host/FlashWorker.o $(REUSED) -framework Foundation -framework Metal -framework IOKit -o $@
cpu: all
	$(BUILD)/splash-flash --cpu-self-test
-include $(BUILD)/host/FlashWorker.d
