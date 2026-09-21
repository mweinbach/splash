# Private candidate only. Compilation/CPU self-test create no backend/device.
# make -f Makefile -f dev/benchmarks/flash_shared_expert_fused_oracle.mk \
#   BUILD=build/flash-shared-expert-fused-v2 flash-shared-expert-fused-oracle
FLASH_SHARED_FUSED_BASE ?= build/flash-default-v5
FLASH_SHARED_FUSED_ORACLE := $(BUILD)/flash-shared-expert-fused-oracle
FLASH_SHARED_FUSED_LIB := $(BUILD)/flash-shared-fused.metallib
FLASH_SHARED_FUSED_AIR := $(BUILD)/flash_shared_expert_fused.air
FLASH_SHARED_FUSED_OBJECT := $(BUILD)/FlashSharedExpertFused.o
FLASH_SHARED_FUSED_PROVENANCE := $(BUILD)/FlashSharedExpertHostProvenance.hpp
FLASH_SHARED_FUSED_AIRS := $(wildcard $(FLASH_SHARED_FUSED_BASE)/metal/*/*.air)
FLASH_SHARED_FUSED_OBJECTS := $(addprefix $(FLASH_SHARED_FUSED_BASE)/flash/, \
	FlashAffine.o FlashMoE.o FlashAffineMPP.o FlashDenseCache.o FlashDescriptor.o FlashWeights.o FlashOperandStore.o) \
	$(FLASH_SHARED_FUSED_BASE)/engine/metal/MetalBackend.o \
	$(FLASH_SHARED_FUSED_BASE)/engine/metal/DeviceCapabilities.o

$(FLASH_SHARED_FUSED_AIR): dev/benchmarks/flash_shared_expert_fused.metal
	mkdir -p $(@D)
	$(METAL) $(PROD_METALFLAGS) -c $< -o $@

$(FLASH_SHARED_FUSED_LIB): $(FLASH_SHARED_FUSED_AIR) $(FLASH_SHARED_FUSED_AIRS)
	@test -n "$(FLASH_SHARED_FUSED_AIRS)" || { echo 'error: compile a Flash base library first'; exit 1; }
	$(METALLIB) $(FLASH_SHARED_FUSED_AIRS) $(FLASH_SHARED_FUSED_AIR) -o $@

$(FLASH_SHARED_FUSED_OBJECT): dev/benchmarks/FlashSharedExpertFused.cpp \
		dev/benchmarks/FlashSharedExpertFused.hpp runtime/metal/MetalBackend.hpp
	mkdir -p $(@D)
	$(CXX) $(ENGINE_CXXFLAGS) -c $< -o $@

$(FLASH_SHARED_FUSED_PROVENANCE): dev/tools/flash_oracle_host_provenance.py \
		runtime/metal/MetalBackend.hpp $(FLASH_SHARED_FUSED_OBJECTS) $(FLASH_SHARED_FUSED_OBJECT)
	$(PYTHON) dev/tools/flash_oracle_host_provenance.py --output $@ \
		--abi-header runtime/metal/MetalBackend.hpp $(FLASH_SHARED_FUSED_OBJECTS) $(FLASH_SHARED_FUSED_OBJECT)

$(FLASH_SHARED_FUSED_ORACLE): dev/benchmarks/flash_shared_expert_fused_oracle.mm \
		dev/benchmarks/FlashSharedExpertFused.hpp $(FLASH_SHARED_FUSED_OBJECT) \
		$(FLASH_SHARED_FUSED_OBJECTS) $(FLASH_SHARED_FUSED_LIB) $(FLASH_SHARED_FUSED_PROVENANCE)
	$(CXX) $(ENGINE_OBJCXXFLAGS) -I$(BUILD) $< $(FLASH_SHARED_FUSED_OBJECT) \
		$(FLASH_SHARED_FUSED_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

.PHONY: flash-shared-expert-fused-oracle
flash-shared-expert-fused-oracle: $(FLASH_SHARED_FUSED_ORACLE)
