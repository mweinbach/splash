BUILD ?= build/hybrid-q4-i8-expert-residency-sep21-v4
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/prefill4k_attention -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
include $(BUILD)/link-inputs.mk
REUSED := $(filter-out %/FlashWeights.o,$(REUSED))
CHANGED_NAMES := FlashMoE FlashMoEBlocked FlashExpertDenseCache FlashInt8ExpertStore FlashFloatDenseCache FlashDenseCache FlashForward
REUSED_CHANGED := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(CHANGED_NAMES)))
BULK := $(BUILD)/host/Prefill4kQSABulk.o $(BUILD)/host/Prefill4kQSACoalesced.o
.PHONY: all cpu-self-test
all: $(BUILD)/splash-flash
$(BUILD)/host/FlashWorker.o: $(BUILD)/source/runtime/flash/FlashWorker.mm $(BUILD)/source/runtime/flash/FlashWeights.hpp
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/FlashWeights.o: $(BUILD)/source/runtime/flash/FlashWeights.mm $(BUILD)/source/runtime/flash/FlashWeights.hpp
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/splash-flash: $(BUILD)/host/FlashWorker.o $(BUILD)/host/FlashWeights.o $(REUSED_CHANGED) $(BULK) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
cpu-self-test: $(BUILD)/splash-flash
	$(BUILD)/policy-cpu
	$(BUILD)/policy-cpu --freeze0
	$(BUILD)/policy-cpu --freeze1
	$(BUILD)/splash-flash --cpu-self-test
-include $(BUILD)/host/FlashWorker.d $(BUILD)/host/FlashWeights.d
