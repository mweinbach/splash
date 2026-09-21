# Fresh production primitive with frozen private dot-witness shader.
# No active or historical artifact is replaced. Root alone executes GPU work.
FLASH_SHARED_PRODUCTION_BASE ?= build/flash-shared-expert-runtime-v1
FLASH_SHARED_PRODUCTION_ORACLE := $(BUILD)/flash-shared-expert-production-oracle
FLASH_SHARED_PRODUCTION_LIB := $(BUILD)/flash-shared-production.metallib
FLASH_SHARED_PRODUCTION_AIR := $(BUILD)/flash_shared_expert_private_witness.air
FLASH_SHARED_PRODUCTION_WITNESS := $(BUILD)/FlashSharedExpertPrivateWitness.o
FLASH_SHARED_PRODUCTION_PROVENANCE := $(BUILD)/FlashSharedExpertHostProvenance.hpp
FLASH_SHARED_PRODUCTION_AIRS := $(wildcard $(FLASH_SHARED_PRODUCTION_BASE)/metal/*/*.air)
FLASH_SHARED_PRODUCTION_OBJECTS := $(addprefix $(FLASH_SHARED_PRODUCTION_BASE)/flash/, \
	FlashAffine.o FlashMoE.o FlashAffineMPP.o FlashDenseCache.o FlashDescriptor.o FlashWeights.o FlashOperandStore.o FlashSharedExpertFused.o) \
	$(FLASH_SHARED_PRODUCTION_BASE)/engine/metal/MetalBackend.o \
	$(FLASH_SHARED_PRODUCTION_BASE)/engine/metal/DeviceCapabilities.o

$(FLASH_SHARED_PRODUCTION_AIR): dev/benchmarks/flash_shared_expert_fused.metal
	mkdir -p $(@D)
	$(METAL) $(PROD_METALFLAGS) -c $< -o $@

$(FLASH_SHARED_PRODUCTION_LIB): $(FLASH_SHARED_PRODUCTION_AIR) $(FLASH_SHARED_PRODUCTION_AIRS)
	@test -n "$(FLASH_SHARED_PRODUCTION_AIRS)" || { echo 'error: compile the production runtime first'; exit 1; }
	$(METALLIB) $(FLASH_SHARED_PRODUCTION_AIRS) $(FLASH_SHARED_PRODUCTION_AIR) -o $@

$(FLASH_SHARED_PRODUCTION_WITNESS): dev/benchmarks/FlashSharedExpertFused.cpp \
		dev/benchmarks/FlashSharedExpertFused.hpp runtime/metal/MetalBackend.hpp
	mkdir -p $(@D)
	$(CXX) $(ENGINE_CXXFLAGS) -c $< -o $@

$(FLASH_SHARED_PRODUCTION_PROVENANCE): dev/tools/flash_oracle_host_provenance.py \
		runtime/metal/MetalBackend.hpp $(FLASH_SHARED_PRODUCTION_OBJECTS) $(FLASH_SHARED_PRODUCTION_WITNESS)
	$(PYTHON) dev/tools/flash_oracle_host_provenance.py --output $@ \
		--abi-header runtime/metal/MetalBackend.hpp $(FLASH_SHARED_PRODUCTION_OBJECTS) $(FLASH_SHARED_PRODUCTION_WITNESS)

$(FLASH_SHARED_PRODUCTION_ORACLE): dev/benchmarks/flash_shared_expert_production_oracle.mm \
		$(FLASH_SHARED_PRODUCTION_WITNESS) $(FLASH_SHARED_PRODUCTION_OBJECTS) \
		$(FLASH_SHARED_PRODUCTION_LIB) $(FLASH_SHARED_PRODUCTION_PROVENANCE)
	$(CXX) $(ENGINE_OBJCXXFLAGS) -I$(BUILD) $< $(FLASH_SHARED_PRODUCTION_WITNESS) \
		$(FLASH_SHARED_PRODUCTION_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

.PHONY: flash-shared-expert-production-oracle
flash-shared-expert-production-oracle: $(FLASH_SHARED_PRODUCTION_ORACLE)
