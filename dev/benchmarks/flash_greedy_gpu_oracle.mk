# Isolated source/CPU build. Root alone runs the GPU oracle.
# make -f Makefile -f dev/benchmarks/flash_greedy_gpu_oracle.mk \
#   BUILD=build/flash-greedy-gpu build/flash-greedy-gpu/flash-greedy-gpu-oracle
FLASH_GREEDY_GPU_ORACLE := $(BUILD)/flash-greedy-gpu-oracle
FLASH_GREEDY_GPU_PRIVATE_LIB := $(BUILD)/flash-greedy-gpu.metallib
FLASH_GREEDY_GPU_OBJECTS := $(FLASH_BUILD)/FlashGreedyGPU.o $(FLASH_CORE_OBJECTS)

$(FLASH_GREEDY_GPU_ORACLE) $(FLASH_BUILD)/FlashGreedyGPU.o \
		$(FLASH_GREEDY_GPU_PRIVATE_LIB): BUILD_CONFIG := $(CONFIG_DIGEST)

$(FLASH_GREEDY_GPU_PRIVATE_LIB): $(METAL_BUILD)/shared/flash_greedy_gpu.air
	$(RUN_CONFIGURED) $(METALLIB) $< -o $@

$(FLASH_GREEDY_GPU_ORACLE): dev/benchmarks/flash_greedy_gpu_oracle.mm \
		runtime/flash/FlashGreedyGPU.hpp runtime/flash/FlashGreedy.hpp \
		runtime/flash/FlashMTPWindow.hpp $(FLASH_GREEDY_GPU_OBJECTS) $(FLASH_GREEDY_GPU_PRIVATE_LIB)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $< \
		$(FLASH_GREEDY_GPU_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

.PHONY: flash-greedy-gpu-oracle
flash-greedy-gpu-oracle: $(FLASH_GREEDY_GPU_ORACLE)
