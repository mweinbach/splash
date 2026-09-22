BUILD ?= build/prefill-qsa-twopass-sep21
SOURCE := $(BUILD)/source
RUNTIME := $(SOURCE)/runtime
BASE := $(SOURCE)/base
EXP := $(SOURCE)/experiment
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
CXXFLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -mmacosx-version-min=27.0 -I$(RUNTIME) -I$(BASE) -I$(EXP)
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -I$(RUNTIME)
OBJECTS := $(addprefix $(BUILD)/host/,MetalBackend.o DeviceCapabilities.o MemoryGovernor.o FlashQSA.o FlashQSAFast.o FlashQSAMPP.o coalesced.o bulk.o twopass.o oracle.o)
AIRS := $(addprefix $(BUILD)/metal/,flash_qsa.air flash_qsa_fast.air flash_qsa_mpp.air flash_qsa_row_tiles.air bulk_attention_sg8.air candidate.air)
.PHONY: all cpu
all: $(BUILD)/oracle $(BUILD)/splash.metallib
$(BUILD)/host $(BUILD)/metal:
	mkdir -p $@
$(BUILD)/host/MetalBackend.o: $(RUNTIME)/metal/MetalBackend.mm | $(BUILD)/host
	$(CXX) $(CXXFLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/host/DeviceCapabilities.o: $(RUNTIME)/metal/DeviceCapabilities.cpp | $(BUILD)/host
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/MemoryGovernor.o: $(RUNTIME)/engine/MemoryGovernor.cpp | $(BUILD)/host
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/Flash%.o: $(RUNTIME)/flash/Flash%.cpp | $(BUILD)/host
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/coalesced.o: $(BASE)/coalesced.cpp | $(BUILD)/host
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/bulk.o: $(BASE)/bulk.cpp | $(BUILD)/host
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/twopass.o: $(EXP)/twopass.cpp | $(BUILD)/host
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@
$(BUILD)/host/oracle.o: $(EXP)/oracle.mm $(EXP)/base_oracle.mm | $(BUILD)/host
	$(CXX) $(CXXFLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(BUILD)/oracle: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -fobjc-arc $^ -framework Foundation -framework Metal -framework IOKit -o $@
$(BUILD)/metal/flash_%.air: $(RUNTIME)/metal/kernels/shared/flash_%.metal | $(BUILD)/metal
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/metal/bulk_attention_sg8.air: $(BASE)/bulk_attention_sg8.metal | $(BUILD)/metal
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/metal/candidate.air: $(EXP)/candidate.metal | $(BUILD)/metal
	$(METAL) $(METALFLAGS) -c $< -o $@
$(BUILD)/splash.metallib: $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
cpu: $(BUILD)/oracle
	$(BUILD)/oracle --cpu-self-test
-include $(OBJECTS:.o=.d)
