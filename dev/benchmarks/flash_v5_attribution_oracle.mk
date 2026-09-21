# Private fresh ABI/normal/stage attribution build; Root alone runs inference.
FLASH_V5_ATTRIBUTION_ORACLE := $(BUILD)/flash-v5-attribution-oracle
$(FLASH_V5_ATTRIBUTION_ORACLE): BUILD_CONFIG := $(CONFIG_DIGEST)
$(FLASH_V5_ATTRIBUTION_ORACLE): dev/benchmarks/flash_v5_attribution_oracle.mm \
        $(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $< \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
		$(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@
.PHONY: flash-v5-attribution-oracle
flash-v5-attribution-oracle: $(FLASH_V5_ATTRIBUTION_ORACLE)
