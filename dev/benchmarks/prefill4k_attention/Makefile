BUILD ?= build/prefill4k-attention
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -Iruntime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0
OBJECTS := $(addprefix $(BUILD)/,MetalBackend.o DeviceCapabilities.o FlashQSA.o FlashQSAFast.o FlashQSAMPP.o oracle.o)
KERNELS := flash_qsa flash_qsa_fast flash_qsa_mpp
AIRS := $(addprefix $(BUILD)/,$(addsuffix .air,$(KERNELS))) $(BUILD)/candidate.air

.PHONY: all cpu-self-test
all: $(BUILD)/oracle $(BUILD)/attention.metallib
$(BUILD):
	mkdir -p $@
$(BUILD)/MetalBackend.o: runtime/metal/MetalBackend.mm | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/DeviceCapabilities.o: runtime/metal/DeviceCapabilities.cpp | $(BUILD)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/Flash%.o: runtime/flash/Flash%.cpp | $(BUILD)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/oracle.o: dev/benchmarks/prefill4k_attention/oracle.mm | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/oracle: $(OBJECTS)
	$(CXX) $(FLAGS) -fobjc-arc $(OBJECTS) -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/%.air: runtime/metal/kernels/shared/%.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/candidate.air: dev/benchmarks/prefill4k_attention/candidate.metal runtime/metal/kernels/shared/flash_qsa_row_tiles.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/attention.metallib: $(AIRS)
	xcrun -sdk macosx metallib $(AIRS) -o $@
cpu-self-test: $(BUILD)/oracle
	$(BUILD)/oracle --cpu-self-test
-include $(OBJECTS:.o=.d)

$(BUILD)/coalesced.o: dev/benchmarks/prefill4k_attention/coalesced.cpp dev/benchmarks/prefill4k_attention/coalesced.hpp | $(BUILD)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/coalesced-cpu-test.o: dev/benchmarks/prefill4k_attention/coalesced_cpu_test.cpp | $(BUILD)
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
$(BUILD)/coalesced-cpu-test: $(BUILD)/coalesced-cpu-test.o $(BUILD)/coalesced.o $(filter-out $(BUILD)/oracle.o,$(OBJECTS))
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
.PHONY: coalesced-cpu-test
coalesced-cpu-test: $(BUILD)/coalesced-cpu-test
	$(BUILD)/coalesced-cpu-test

$(BUILD)/query_reuse.air: dev/benchmarks/prefill4k_attention/query_reuse.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/values_n128.air: dev/benchmarks/prefill4k_attention/values_n128.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/staging.metallib: $(AIRS) $(BUILD)/query_reuse.air $(BUILD)/values_n128.air
	xcrun -sdk macosx metallib $^ -o $@
$(BUILD)/query-reuse-cpu-test: dev/benchmarks/prefill4k_attention/query_reuse_cpu_test.cpp | $(BUILD)
	$(CXX) $(FLAGS) $< -o $@
.PHONY: staging query-reuse-cpu-test
staging: $(BUILD)/oracle $(BUILD)/staging.metallib
query-reuse-cpu-test: $(BUILD)/query-reuse-cpu-test
	$(BUILD)/query-reuse-cpu-test
-include $(BUILD)/coalesced.d $(BUILD)/coalesced-cpu-test.d

PV_REGISTER_AIRS := $(addprefix $(BUILD)/,register_pv.air register_pv_strict.air relaxed_pv.air reduce_debug.air)
$(BUILD)/register_pv.air: dev/benchmarks/prefill4k_attention/register_pv.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/register_pv_strict.air: dev/benchmarks/prefill4k_attention/register_pv_strict.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/relaxed_pv.air: dev/benchmarks/prefill4k_attention/relaxed_pv.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/reduce_debug.air: dev/benchmarks/prefill4k_attention/reduce_debug.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/actual-pv.metallib: $(AIRS) $(PV_REGISTER_AIRS)
	xcrun -sdk macosx metallib $^ -o $@
$(BUILD)/actual-oracle.o: dev/benchmarks/prefill4k_attention/actual_oracle.mm dev/benchmarks/prefill4k_attention/oracle.mm | $(BUILD)
	$(CXX) $(FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/actual-oracle: $(BUILD)/actual-oracle.o $(filter-out $(BUILD)/oracle.o,$(OBJECTS))
	$(CXX) $(FLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/register-pv-cpu-test: dev/benchmarks/prefill4k_attention/register_pv_cpu_test.cpp | $(BUILD)
	$(CXX) $(FLAGS) $< -o $@
.PHONY: actual-pv register-pv-cpu-test
actual-pv: $(BUILD)/actual-oracle $(BUILD)/actual-pv.metallib
register-pv-cpu-test: $(BUILD)/register-pv-cpu-test
	$(BUILD)/register-pv-cpu-test
-include $(BUILD)/actual-oracle.d
