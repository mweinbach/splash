BUILD ?= build/sep21-memory-payload
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
HERE := dev/benchmarks/decode_target_sep21
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Iruntime -I$(HERE) -mmacosx-version-min=27.0
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(HERE) -mmacosx-version-min=27.0
OBJECTS := $(BUILD)/MetalBackend.o $(BUILD)/DeviceCapabilities.o $(BUILD)/bandwidth.o
.PHONY: all cpu-self-test
all: $(BUILD)/bandwidth $(BUILD)/bandwidth.metallib
$(BUILD):
	mkdir -p $@
$(BUILD)/MetalBackend.o: runtime/metal/MetalBackend.mm | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/DeviceCapabilities.o: runtime/metal/DeviceCapabilities.cpp | $(BUILD)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/bandwidth.o: $(HERE)/bandwidth.mm $(HERE)/bandwidth.hpp | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/bandwidth: $(OBJECTS)
	$(CXX) $(FLAGS) -fobjc-arc $(OBJECTS) -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/bandwidth.air: $(HERE)/bandwidth.metal $(HERE)/bandwidth.hpp | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/bandwidth.metallib: $(BUILD)/bandwidth.air
	xcrun -sdk macosx metallib $< -o $@
cpu-self-test: $(BUILD)/bandwidth
	$(BUILD)/bandwidth --cpu-self-test
-include $(OBJECTS:.o=.d)
