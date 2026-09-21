# Fresh private HC-down screen. Root alone executes GPU work.
FLASH_HC_DOWN_F32_BASE ?= build/flash-default-v6
FLASH_HC_DOWN_F32_ORACLE := $(BUILD)/flash-hc-down-f32-oracle
FLASH_HC_DOWN_F32_LIB := $(BUILD)/flash-hc-down-f32.metallib
FLASH_HC_DOWN_F32_AIR := $(BUILD)/flash_hc_down_f32_candidate.air
FLASH_HC_DOWN_F32_OBJECT := $(BUILD)/FlashHCDownF32Candidate.o
FLASH_HC_DOWN_F32_PROVENANCE := $(BUILD)/FlashHCDownHostProvenance.hpp
FLASH_HC_DOWN_F32_AIRS := $(wildcard $(FLASH_HC_DOWN_F32_BASE)/metal/*/*.air)
FLASH_HC_DOWN_F32_OBJECTS := $(addprefix $(FLASH_HC_DOWN_F32_BASE)/flash/, \
	FlashAffine.o FlashHC.o FlashHCFused.o FlashFloatDenseCache.o FlashOperandStore.o \
	FlashDenseCache.o FlashAffineMPP.o \
	FlashDescriptor.o FlashWeights.o) \
	$(FLASH_HC_DOWN_F32_BASE)/engine/metal/MetalBackend.o \
	$(FLASH_HC_DOWN_F32_BASE)/engine/metal/DeviceCapabilities.o

$(FLASH_HC_DOWN_F32_AIR): dev/benchmarks/flash_hc_down_f32_candidate.metal \
		dev/benchmarks/FlashHCDownF32CandidateABI.h
	mkdir -p $(@D)
	$(METAL) $(PROD_METALFLAGS) -Idev/benchmarks -c $< -o $@

$(FLASH_HC_DOWN_F32_LIB): $(FLASH_HC_DOWN_F32_AIR) $(FLASH_HC_DOWN_F32_AIRS)
	$(METALLIB) $(FLASH_HC_DOWN_F32_AIRS) $(FLASH_HC_DOWN_F32_AIR) -o $@

$(FLASH_HC_DOWN_F32_OBJECT): dev/benchmarks/FlashHCDownF32Candidate.cpp \
		dev/benchmarks/FlashHCDownF32Candidate.hpp dev/benchmarks/FlashHCDownF32CandidateABI.h \
		runtime/metal/MetalBackend.hpp
	mkdir -p $(@D)
	$(CXX) $(ENGINE_CXXFLAGS) -c $< -o $@

$(FLASH_HC_DOWN_F32_PROVENANCE): dev/tools/flash_oracle_host_provenance.py \
		runtime/metal/MetalBackend.hpp $(FLASH_HC_DOWN_F32_OBJECTS) $(FLASH_HC_DOWN_F32_OBJECT)
	$(PYTHON) dev/tools/flash_oracle_host_provenance.py --output $@ \
		--abi-header runtime/metal/MetalBackend.hpp $(FLASH_HC_DOWN_F32_OBJECTS) $(FLASH_HC_DOWN_F32_OBJECT)

$(FLASH_HC_DOWN_F32_ORACLE): dev/benchmarks/flash_hc_down_f32_candidate_oracle.mm \
		$(FLASH_HC_DOWN_F32_OBJECT) $(FLASH_HC_DOWN_F32_OBJECTS) \
		$(FLASH_HC_DOWN_F32_LIB) $(FLASH_HC_DOWN_F32_PROVENANCE)
	$(CXX) $(ENGINE_OBJCXXFLAGS) -I$(BUILD) $< $(FLASH_HC_DOWN_F32_OBJECT) \
		$(FLASH_HC_DOWN_F32_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

.PHONY: flash-hc-down-f32-oracle
flash-hc-down-f32-oracle: $(FLASH_HC_DOWN_F32_ORACLE)
