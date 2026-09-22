BUILD ?= build/prefill-hc-inject-norm-sep21
ROOT ?= .
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -I$(ROOT)/runtime -I$(ROOT) -mmacosx-version-min=27.0 -ffp-contract=off
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(ROOT)/runtime -mmacosx-version-min=27.0
HERE := dev/benchmarks/prefill_hc_inject_norm_sep21
OBJECTS := $(addprefix $(BUILD)/,MetalBackend.o DeviceCapabilities.o FlashHC.o oracle.o)
AIRS := $(BUILD)/flash_hc.air $(BUILD)/candidate.air
.PHONY: all cpu-self-test
all: $(BUILD)/oracle $(BUILD)/splash.metallib
$(BUILD):
	mkdir -p $@
$(BUILD)/MetalBackend.o: $(ROOT)/runtime/metal/MetalBackend.mm | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/DeviceCapabilities.o: $(ROOT)/runtime/metal/DeviceCapabilities.cpp | $(BUILD)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/FlashHC.o: $(ROOT)/runtime/flash/FlashHC.cpp | $(BUILD)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/oracle.o: $(ROOT)/$(HERE)/oracle.mm $(ROOT)/$(HERE)/bridge.hpp | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/oracle: $(OBJECTS)
	$(CXX) $(FLAGS) -fobjc-arc $(OBJECTS) -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/flash_hc.air: $(ROOT)/runtime/metal/kernels/shared/flash_hc.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/candidate.air: $(ROOT)/$(HERE)/candidate.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/splash.metallib: $(AIRS)
	xcrun -sdk macosx metallib $(AIRS) -o $@
cpu-self-test: $(BUILD)/oracle
	$(BUILD)/oracle --cpu-self-test
-include $(OBJECTS:.o=.d)
