BUILD ?= build/ultra-quantized-locality
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0
OBJECTS := $(BUILD)/MetalBackend.o $(BUILD)/DeviceCapabilities.o $(BUILD)/oracle.o
.PHONY: all cpu-self-test
all: $(BUILD)/oracle $(BUILD)/locality.metallib
$(BUILD):
	mkdir -p $@
$(BUILD)/MetalBackend.o: runtime/metal/MetalBackend.mm | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/DeviceCapabilities.o: runtime/metal/DeviceCapabilities.cpp | $(BUILD)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/oracle.o: dev/benchmarks/ultra_quantized_locality/oracle.mm | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/oracle: $(OBJECTS)
	$(CXX) $(FLAGS) -fobjc-arc $(OBJECTS) -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/candidate.air: dev/benchmarks/ultra_quantized_locality/candidate.metal runtime/metal/kernels/shared/flash_affine.metal runtime/metal/kernels/shared/flash_expert_qmv.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/locality.metallib: $(BUILD)/candidate.air
	xcrun -sdk macosx metallib $< -o $@
cpu-self-test: $(BUILD)/oracle
	$(BUILD)/oracle --cpu-self-test
-include $(OBJECTS:.o=.d)
