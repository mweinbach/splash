ROOT := $(abspath .)
BUILD ?= build/idle-minimal-resource-probe-v1
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
OBJCFLAGS := $(FLAGS) -fobjc-arc
OBJECTS := $(BUILD)/MetalBackend.o $(BUILD)/DeviceCapabilities.o $(BUILD)/FlashDescriptor.o $(BUILD)/FlashWeights.o $(BUILD)/oracle.o

.PHONY: all
all: $(BUILD)/oracle $(BUILD)/splash.metallib

$(BUILD):
	mkdir -p $@

$(BUILD)/MetalBackend.o: runtime/metal/MetalBackend.mm | $(BUILD)
	$(CXX) $(OBJCFLAGS) -MMD -MP -c $< -o $@
$(BUILD)/DeviceCapabilities.o: runtime/metal/DeviceCapabilities.cpp | $(BUILD)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/FlashDescriptor.o: runtime/flash/FlashDescriptor.mm | $(BUILD)
	$(CXX) $(OBJCFLAGS) -MMD -MP -c $< -o $@
$(BUILD)/FlashWeights.o: runtime/flash/FlashWeights.mm | $(BUILD)
	$(CXX) $(OBJCFLAGS) -MMD -MP -c $< -o $@
$(BUILD)/oracle.o: dev/benchmarks/idle_minimal_resource_probe/oracle.mm | $(BUILD)
	$(CXX) $(OBJCFLAGS) -MMD -MP -c $< -o $@
$(BUILD)/oracle: $(OBJECTS)
	$(CXX) $(OBJCFLAGS) $(OBJECTS) -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/probe.air: dev/benchmarks/idle_minimal_resource_probe/probe.metal | $(BUILD)
	xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -c $< -o $@
$(BUILD)/splash.metallib: $(BUILD)/probe.air
	xcrun -sdk macosx metallib $< -o $@

-include $(OBJECTS:.o=.d)
