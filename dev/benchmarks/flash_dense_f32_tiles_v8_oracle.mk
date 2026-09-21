# Private original-F32 dense tile screen; Root alone executes GPU work.
FLASH_DENSE_F32_V8_BASE ?= build/flash-default-v7-instrumented
FLASH_DENSE_F32_V8_ORACLE := $(BUILD)/flash-dense-f32-tiles-v8-oracle
FLASH_DENSE_F32_V8_LIB := $(BUILD)/flash-dense-f32-tiles-v8.metallib
FLASH_DENSE_F32_V8_AIR := $(BUILD)/flash_dense_f32_tiles_v8.air
FLASH_DENSE_F32_V8_OBJECT := $(BUILD)/FlashDenseF32TilesV8.o
FLASH_DENSE_F32_V8_PROVENANCE := $(BUILD)/FlashDenseF32TilesV8HostProvenance.hpp
FLASH_DENSE_F32_V8_AIRS := $(wildcard $(FLASH_DENSE_F32_V8_BASE)/metal/*/*.air)
FLASH_DENSE_F32_V8_OBJECTS := $(addprefix $(FLASH_DENSE_F32_V8_BASE)/flash/, \
	FlashAffine.o FlashFloatDenseCache.o FlashOperandStore.o FlashDenseCache.o \
	FlashDescriptor.o FlashWeights.o) \
	$(FLASH_DENSE_F32_V8_BASE)/engine/metal/MetalBackend.o \
	$(FLASH_DENSE_F32_V8_BASE)/engine/metal/DeviceCapabilities.o

$(FLASH_DENSE_F32_V8_AIR): dev/benchmarks/flash_dense_f32_tiles_v8.metal
	mkdir -p $(@D)
	$(METAL) $(PROD_METALFLAGS) -c $< -o $@

$(FLASH_DENSE_F32_V8_LIB): $(FLASH_DENSE_F32_V8_AIR) $(FLASH_DENSE_F32_V8_AIRS)
	$(METALLIB) $(FLASH_DENSE_F32_V8_AIRS) $(FLASH_DENSE_F32_V8_AIR) -o $@

$(FLASH_DENSE_F32_V8_OBJECT): dev/benchmarks/FlashDenseF32TilesV8.cpp \
		dev/benchmarks/FlashDenseF32TilesV8.hpp runtime/metal/MetalBackend.hpp
	mkdir -p $(@D)
	$(CXX) $(ENGINE_CXXFLAGS) -c $< -o $@

$(FLASH_DENSE_F32_V8_PROVENANCE): dev/tools/flash_oracle_host_provenance.py \
		runtime/metal/MetalBackend.hpp $(FLASH_DENSE_F32_V8_OBJECTS) $(FLASH_DENSE_F32_V8_OBJECT)
	$(PYTHON) dev/tools/flash_oracle_host_provenance.py --output $@ \
		--abi-header runtime/metal/MetalBackend.hpp $(FLASH_DENSE_F32_V8_OBJECTS) $(FLASH_DENSE_F32_V8_OBJECT)

$(FLASH_DENSE_F32_V8_ORACLE): dev/benchmarks/flash_dense_f32_tiles_v8_oracle.mm \
		$(FLASH_DENSE_F32_V8_OBJECT) $(FLASH_DENSE_F32_V8_OBJECTS) \
		$(FLASH_DENSE_F32_V8_LIB) $(FLASH_DENSE_F32_V8_PROVENANCE)
	$(CXX) $(ENGINE_OBJCXXFLAGS) -ffp-contract=off -I$(BUILD) $< \
		$(FLASH_DENSE_F32_V8_OBJECT) $(FLASH_DENSE_F32_V8_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

.PHONY: flash-dense-f32-tiles-v8-oracle
flash-dense-f32-tiles-v8-oracle: $(FLASH_DENSE_F32_V8_ORACLE)
