BUILD ?= build/ultra-dense-locality
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0
OBJECTS := $(BUILD)/MetalBackend.o $(BUILD)/DeviceCapabilities.o $(BUILD)/oracle.o
KERNELS := flash_dense_cache flash_dense_cache_m64 flash_dense_small_rows flash_affine
AIRS := $(addprefix $(BUILD)/,$(addsuffix .air,$(KERNELS))) $(BUILD)/candidate.air

.PHONY: all cpu-self-test policy-test
all: $(BUILD)/oracle $(BUILD)/locality.metallib

$(BUILD):
	mkdir -p $@
$(BUILD)/MetalBackend.o: runtime/metal/MetalBackend.mm | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/DeviceCapabilities.o: runtime/metal/DeviceCapabilities.cpp | $(BUILD)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/oracle.o: dev/benchmarks/ultra_dense_locality/oracle.mm | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/oracle: $(OBJECTS)
	$(CXX) $(FLAGS) -fobjc-arc $(OBJECTS) -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/%.air: runtime/metal/kernels/shared/%.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/candidate.air: dev/benchmarks/ultra_dense_locality/candidate.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/locality.metallib: $(AIRS)
	xcrun -sdk macosx metallib $(AIRS) -o $@
cpu-self-test: $(BUILD)/oracle
	$(BUILD)/oracle --cpu-self-test
$(BUILD)/policy-test: dev/benchmarks/ultra_dense_locality/policy_test.cpp \
    runtime/flash/FlashDenseTraversal.hpp runtime/metal/abi/FlashDenseCache.h | $(BUILD)
	$(CXX) $(FLAGS) $< -o $@
policy-test: $(BUILD)/policy-test
	$(BUILD)/policy-test

-include $(OBJECTS:.o=.d)
