# Private primitive oracle. Compilation and --cpu-self-test submit no GPU work.
# Root alone runs the GPU qualification.
# make -f Makefile -f dev/benchmarks/flash_gpu_prefill_copy_oracle.mk \
#   BUILD=build/flash-gpu-prefill-copy/primitive flash-gpu-prefill-copy-oracle
FLASH_GPU_PREFILL_COPY_ORACLE := $(BUILD)/flash-gpu-prefill-copy-oracle
FLASH_GPU_PREFILL_COPY_PRIVATE_LIB := $(BUILD)/flash-gpu-prefill-copy.metallib
FLASH_GPU_PREFILL_COPY_OBJECTS := $(ENGINE_BUILD)/metal/MetalBackend.o \
	$(ENGINE_BUILD)/metal/DeviceCapabilities.o

$(FLASH_GPU_PREFILL_COPY_ORACLE) $(FLASH_GPU_PREFILL_COPY_PRIVATE_LIB): \
		BUILD_CONFIG := $(CONFIG_DIGEST)

$(FLASH_GPU_PREFILL_COPY_PRIVATE_LIB): $(METAL_BUILD)/shared/flash_forward.air
	$(RUN_CONFIGURED) $(METALLIB) $< -o $@

$(FLASH_GPU_PREFILL_COPY_ORACLE): dev/benchmarks/flash_gpu_prefill_copy_oracle.mm \
		runtime/flash/FlashBatchPrefill.hpp runtime/metal/abi/FlashForward.h \
		$(FLASH_GPU_PREFILL_COPY_OBJECTS) $(FLASH_GPU_PREFILL_COPY_PRIVATE_LIB)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $< \
		$(FLASH_GPU_PREFILL_COPY_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

.PHONY: flash-gpu-prefill-copy-oracle
flash-gpu-prefill-copy-oracle: $(FLASH_GPU_PREFILL_COPY_ORACLE)
