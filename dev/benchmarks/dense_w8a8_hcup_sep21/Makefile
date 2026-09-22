BUILD ?= build/dense-w8a8-hcup-sep21
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -ffp-contract=off
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0
OBJECTS := $(BUILD)/MetalBackend.o $(BUILD)/DeviceCapabilities.o $(BUILD)/oracle.o
AIRS := $(BUILD)/flash_dense_cache.air $(BUILD)/flash_hc.air $(BUILD)/candidate.air $(BUILD)/post_probe.air
.PHONY: all cpu-self-test preview freeze
all: $(BUILD)/oracle $(BUILD)/hcup.metallib
$(BUILD):
	mkdir -p $@
$(BUILD)/MetalBackend.o: runtime/metal/MetalBackend.mm | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/DeviceCapabilities.o: runtime/metal/DeviceCapabilities.cpp | $(BUILD)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/oracle.o: dev/benchmarks/dense_w8a8_hcup_sep21/oracle.mm | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/oracle: $(OBJECTS)
	$(CXX) $(FLAGS) -fobjc-arc $(OBJECTS) -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/flash_dense_cache.air: runtime/metal/kernels/shared/flash_dense_cache.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/flash_hc.air: runtime/metal/kernels/shared/flash_hc.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/candidate.air: dev/benchmarks/dense_w8a8_hcup_sep21/candidate.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/post_probe.air: dev/benchmarks/dense_w8a8_hcup_sep21/post_probe.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/hcup.metallib: $(AIRS)
	xcrun -sdk macosx metallib $(AIRS) -o $@
cpu-self-test: $(BUILD)/oracle
	$(BUILD)/oracle --cpu-self-test
preview: $(BUILD)/oracle
	$(BUILD)/oracle --preview
$(BUILD)/candidate.ll: dev/benchmarks/dense_w8a8_hcup_sep21/candidate.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -S -emit-llvm $< -o $@
freeze: all $(BUILD)/candidate.ll
	python3 dev/benchmarks/dense_w8a8_hcup_sep21/freeze.py
-include $(OBJECTS:.o=.d)
