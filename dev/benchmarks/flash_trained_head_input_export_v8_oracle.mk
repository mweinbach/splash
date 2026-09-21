# Private original-head clone exposes completed shared input; no GPU edits.
FLASH_HEAD_INPUT_EXPORT := $(BUILD)/flash-trained-head-input-export-v8-oracle
FLASH_HEAD_INPUT_SNAPSHOT := $(BUILD)/snapshot
ENGINE_CXXFLAGS := -I$(FLASH_HEAD_INPUT_SNAPSHOT) $(ENGINE_CXXFLAGS)
ENGINE_OBJCXXFLAGS := -I$(FLASH_HEAD_INPUT_SNAPSHOT) $(ENGINE_OBJCXXFLAGS)
$(FLASH_BUILD)/FlashMTP.o: $(FLASH_HEAD_INPUT_SNAPSHOT)/flash/FlashMTP.cpp $(FLASH_HEAD_INPUT_SNAPSHOT)/flash/FlashMTP.hpp
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_CXXFLAGS) -MMD -MP -c $< -o $@
$(FLASH_HEAD_INPUT_EXPORT): BUILD_CONFIG := $(CONFIG_DIGEST)
$(FLASH_HEAD_INPUT_EXPORT): dev/benchmarks/flash_trained_head_input_export_v8_oracle.mm \
        $(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $< \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
		$(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@
.PHONY: flash-trained-head-input-export-v8-oracle
flash-trained-head-input-export-v8-oracle: $(FLASH_HEAD_INPUT_EXPORT)
