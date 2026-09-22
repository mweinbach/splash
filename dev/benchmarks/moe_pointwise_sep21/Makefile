BUILD ?= build/moe-pointwise-sep21
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0
OBJECTS := $(BUILD)/MetalBackend.o $(BUILD)/DeviceCapabilities.o $(BUILD)/oracle.o
AIRS := $(BUILD)/flash_moe.air $(BUILD)/candidate.air
.PHONY: all cpu-self-test
all: $(BUILD)/oracle $(BUILD)/pointwise.metallib
$(BUILD):
	mkdir -p $@
$(BUILD)/MetalBackend.o: runtime/metal/MetalBackend.mm | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/DeviceCapabilities.o: runtime/metal/DeviceCapabilities.cpp | $(BUILD)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/oracle.o: dev/benchmarks/moe_pointwise_sep21/oracle.mm | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/oracle: $(OBJECTS)
	$(CXX) $(FLAGS) -fobjc-arc $(OBJECTS) -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/flash_moe.air: runtime/metal/kernels/shared/flash_moe.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/candidate.air: dev/benchmarks/moe_pointwise_sep21/candidate.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/pointwise.metallib: $(AIRS)
	xcrun -sdk macosx metallib $(AIRS) -o $@
cpu-self-test: $(BUILD)/oracle
	$(BUILD)/oracle --cpu-self-test
-include $(OBJECTS:.o=.d)
