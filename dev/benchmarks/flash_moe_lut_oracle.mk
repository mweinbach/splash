# Isolated saved-LUT candidate. No production library or binary is replaced.
# Compilation, CPU self-tests and sidecar validation submit no GPU work.
# Root alone runs inference/Metal qualification.
# make -f Makefile -f dev/benchmarks/flash_moe_lut_oracle.mk \
#   BUILD=build/flash-expert-lut-format flash-moe-lut-oracle
FLASH_LUT_BASE ?= build/flash-next
FLASH_LUT_ORACLE := $(BUILD)/flash-moe-lut-oracle
FLASH_LUT_PRIVATE_LIB := $(BUILD)/flash-lut.metallib
FLASH_LUT_AIR := $(BUILD)/flash_moe_lut.air
FLASH_LUT_SIDE_OBJECT := $(BUILD)/FlashExpertLUTSidecar.o
FLASH_LUT_BASE_AIRS := $(wildcard $(FLASH_LUT_BASE)/metal/*/*.air)
FLASH_LUT_OBJECTS := $(addprefix $(FLASH_LUT_BASE)/flash/, \
	FlashAffine.o FlashMoE.o FlashMoEBuckets.o FlashMoEBlocked.o \
	FlashDescriptor.o FlashWeights.o) \
	$(FLASH_LUT_BASE)/engine/metal/MetalBackend.o \
	$(FLASH_LUT_BASE)/engine/metal/DeviceCapabilities.o

$(FLASH_LUT_AIR): dev/benchmarks/flash_moe_lut.metal
	mkdir -p $(@D)
	$(METAL) $(PROD_METALFLAGS) -c $< -o $@

$(FLASH_LUT_PRIVATE_LIB): $(FLASH_LUT_AIR) $(FLASH_LUT_BASE_AIRS)
	@test -n "$(FLASH_LUT_BASE_AIRS)" || { echo 'error: compile a Flash base library first'; exit 1; }
	$(METALLIB) $(FLASH_LUT_BASE_AIRS) $(FLASH_LUT_AIR) -o $@

$(FLASH_LUT_SIDE_OBJECT): dev/benchmarks/FlashExpertLUTSidecar.mm \
		dev/benchmarks/FlashExpertLUTSidecar.hpp
	mkdir -p $(@D)
	$(CXX) $(ENGINE_OBJCXXFLAGS) -c $< -o $@

$(FLASH_LUT_ORACLE): dev/benchmarks/flash_moe_lut_oracle.mm \
		dev/benchmarks/FlashExpertLUTSidecar.hpp $(FLASH_LUT_SIDE_OBJECT) \
		$(FLASH_LUT_OBJECTS) $(FLASH_LUT_PRIVATE_LIB)
	$(CXX) $(ENGINE_OBJCXXFLAGS) $< $(FLASH_LUT_SIDE_OBJECT) \
		$(FLASH_LUT_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

.PHONY: flash-moe-lut-oracle
flash-moe-lut-oracle: $(FLASH_LUT_ORACLE)
