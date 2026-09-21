# Private original Q8 vocabulary R1 exact arithmetic geometry screen.
FLASH_HEAD_Q8_R1_V8_BASE ?= build/flash-default-v7-instrumented
FLASH_HEAD_Q8_R1_V8_ORACLE := $(BUILD)/flash-head-q8-r1-v8-oracle
FLASH_HEAD_Q8_R1_V8_LIB := $(BUILD)/flash-head-q8-r1-v8.metallib
FLASH_HEAD_Q8_R1_V8_AIR := $(BUILD)/flash_head_q8_r1_v8.air
FLASH_HEAD_Q8_R1_V8_PROVENANCE := $(BUILD)/FlashHeadQ8R1V8HostProvenance.hpp
FLASH_HEAD_Q8_R1_V8_AIRS := $(wildcard $(FLASH_HEAD_Q8_R1_V8_BASE)/metal/*/*.air)
FLASH_HEAD_Q8_R1_V8_OBJECTS := $(addprefix $(FLASH_HEAD_Q8_R1_V8_BASE)/flash/,FlashAffine.o FlashGreedyGPU.o FlashDescriptor.o FlashWeights.o) $(FLASH_HEAD_Q8_R1_V8_BASE)/engine/metal/MetalBackend.o $(FLASH_HEAD_Q8_R1_V8_BASE)/engine/metal/DeviceCapabilities.o
$(FLASH_HEAD_Q8_R1_V8_AIR): dev/benchmarks/flash_head_q8_r1_v8.metal
	mkdir -p $(@D)
	$(METAL) $(PROD_METALFLAGS) -c $< -o $@
$(FLASH_HEAD_Q8_R1_V8_LIB): $(FLASH_HEAD_Q8_R1_V8_AIR) $(FLASH_HEAD_Q8_R1_V8_AIRS)
	$(METALLIB) $(FLASH_HEAD_Q8_R1_V8_AIRS) $(FLASH_HEAD_Q8_R1_V8_AIR) -o $@
$(FLASH_HEAD_Q8_R1_V8_PROVENANCE): dev/tools/flash_oracle_host_provenance.py runtime/metal/MetalBackend.hpp $(FLASH_HEAD_Q8_R1_V8_OBJECTS)
	$(PYTHON) dev/tools/flash_oracle_host_provenance.py --output $@ --abi-header runtime/metal/MetalBackend.hpp $(FLASH_HEAD_Q8_R1_V8_OBJECTS)
$(FLASH_HEAD_Q8_R1_V8_ORACLE): dev/benchmarks/flash_head_q8_r1_v8_oracle.mm $(FLASH_HEAD_Q8_R1_V8_OBJECTS) $(FLASH_HEAD_Q8_R1_V8_PROVENANCE) $(FLASH_HEAD_Q8_R1_V8_LIB)
	$(CXX) $(ENGINE_OBJCXXFLAGS) -ffp-contract=off -I$(BUILD) $< $(FLASH_HEAD_Q8_R1_V8_OBJECTS) $(ENGINE_LINKFLAGS) -o $@
.PHONY: flash-head-q8-r1-v8-oracle
flash-head-q8-r1-v8-oracle: $(FLASH_HEAD_Q8_R1_V8_ORACLE)
