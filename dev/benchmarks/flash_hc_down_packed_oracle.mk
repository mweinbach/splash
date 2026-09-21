# Private packed HCdown experiment. Root alone executes GPU work.
FLASH_HC_DOWN_PACKED_BASE ?= build/flash-hc-lazy-combined-v1
FLASH_HC_DOWN_PACKED_ORACLE := $(BUILD)/flash-hc-down-packed-oracle
FLASH_HC_DOWN_PACKED_LIB := $(BUILD)/flash-hc-down-packed.metallib
FLASH_HC_DOWN_PACKED_AIR := $(BUILD)/flash_hc_down_packed.air
FLASH_HC_DOWN_PACKED_OBJECT := $(BUILD)/FlashHCDownPacked.o
FLASH_HC_DOWN_PACKED_PROVENANCE := $(BUILD)/FlashHCDownPackedHostProvenance.hpp
FLASH_HC_DOWN_PACKED_AIRS := $(wildcard $(FLASH_HC_DOWN_PACKED_BASE)/metal/*/*.air)
FLASH_HC_DOWN_PACKED_OBJECTS := $(addprefix $(FLASH_HC_DOWN_PACKED_BASE)/flash/, \
	FlashAffine.o FlashHC.o FlashHCFused.o FlashFloatDenseCache.o FlashOperandStore.o \
	FlashDenseCache.o FlashAffineMPP.o FlashDescriptor.o FlashWeights.o) \
	$(FLASH_HC_DOWN_PACKED_BASE)/engine/metal/MetalBackend.o \
	$(FLASH_HC_DOWN_PACKED_BASE)/engine/metal/DeviceCapabilities.o

$(FLASH_HC_DOWN_PACKED_AIR): dev/benchmarks/flash_hc_down_packed.metal \
		dev/benchmarks/FlashHCDownPackedABI.h
	mkdir -p $(@D)
	$(METAL) $(PROD_METALFLAGS) -Idev/benchmarks -c $< -o $@

$(FLASH_HC_DOWN_PACKED_LIB): $(FLASH_HC_DOWN_PACKED_AIR) $(FLASH_HC_DOWN_PACKED_AIRS)
	$(METALLIB) $(FLASH_HC_DOWN_PACKED_AIRS) $(FLASH_HC_DOWN_PACKED_AIR) -o $@

$(FLASH_HC_DOWN_PACKED_OBJECT): dev/benchmarks/FlashHCDownPacked.cpp \
		dev/benchmarks/FlashHCDownPacked.hpp dev/benchmarks/FlashHCDownPackedABI.h \
		runtime/metal/MetalBackend.hpp
	mkdir -p $(@D)
	$(CXX) $(ENGINE_CXXFLAGS) -c $< -o $@

$(FLASH_HC_DOWN_PACKED_PROVENANCE): dev/tools/flash_oracle_host_provenance.py \
		runtime/metal/MetalBackend.hpp $(FLASH_HC_DOWN_PACKED_OBJECTS) $(FLASH_HC_DOWN_PACKED_OBJECT)
	$(PYTHON) dev/tools/flash_oracle_host_provenance.py --output $@ \
		--abi-header runtime/metal/MetalBackend.hpp $(FLASH_HC_DOWN_PACKED_OBJECTS) $(FLASH_HC_DOWN_PACKED_OBJECT)

$(FLASH_HC_DOWN_PACKED_ORACLE): dev/benchmarks/flash_hc_down_packed_oracle.mm \
		$(FLASH_HC_DOWN_PACKED_OBJECT) $(FLASH_HC_DOWN_PACKED_OBJECTS) \
		$(FLASH_HC_DOWN_PACKED_LIB) $(FLASH_HC_DOWN_PACKED_PROVENANCE)
	$(CXX) $(ENGINE_OBJCXXFLAGS) -I$(BUILD) $< $(FLASH_HC_DOWN_PACKED_OBJECT) \
		$(FLASH_HC_DOWN_PACKED_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

.PHONY: flash-hc-down-packed-oracle
flash-hc-down-packed-oracle: $(FLASH_HC_DOWN_PACKED_ORACLE)
