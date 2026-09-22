BUILD ?= build/dense-kloop-sync-sep21
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0
OBJECTS := $(BUILD)/MetalBackend.o $(BUILD)/DeviceCapabilities.o $(BUILD)/oracle.o
KERNELS := flash_dense_cache flash_dense_cache_m64 flash_dense_cache_prefill
AIRS := $(addprefix $(BUILD)/,$(addsuffix .air,$(KERNELS))) $(BUILD)/candidate.air

.PHONY: all cpu-self-test preview freeze
all: $(BUILD)/oracle $(BUILD)/kloop.metallib
$(BUILD):
	mkdir -p $@
$(BUILD)/MetalBackend.o: runtime/metal/MetalBackend.mm | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/DeviceCapabilities.o: runtime/metal/DeviceCapabilities.cpp | $(BUILD)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/oracle.o: dev/benchmarks/dense_kloop_sync_sep21/oracle.mm | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/oracle: $(OBJECTS)
	$(CXX) $(FLAGS) -fobjc-arc $(OBJECTS) -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/%.air: runtime/metal/kernels/shared/%.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/candidate.air: dev/benchmarks/dense_kloop_sync_sep21/candidate.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/kloop.metallib: $(AIRS)
	xcrun -sdk macosx metallib $(AIRS) -o $@
cpu-self-test: $(BUILD)/oracle
	$(BUILD)/oracle --cpu-self-test
preview: $(BUILD)/oracle
	$(BUILD)/oracle --preview
freeze: all
	python3 dev/benchmarks/dense_kloop_sync_sep21/freeze.py --build $(BUILD)
-include $(OBJECTS:.o=.d)
