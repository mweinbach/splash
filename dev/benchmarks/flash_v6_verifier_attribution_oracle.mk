# Private source snapshot adds diagnostic route copying, with no production edits.
FLASH_V6_VERIFY_ORACLE := $(BUILD)/flash-v6-verifier-attribution-oracle
FLASH_V6_SNAPSHOT := $(BUILD)/snapshot
FLASH_V6_OBJECTS := $(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS))
$(FLASH_V6_VERIFY_ORACLE) $(FLASH_BUILD)/FlashBatchVerify.o: BUILD_CONFIG := $(CONFIG_DIGEST)
$(FLASH_BUILD)/FlashBatchVerify.o: $(FLASH_V6_SNAPSHOT)/flash/FlashBatchVerify.cpp \
        $(FLASH_V6_SNAPSHOT)/flash/FlashBatchVerify.hpp | $(FLASH_BUILD)
	$(RUN_CONFIGURED) $(CXX) -I$(FLASH_V6_SNAPSHOT) $(ENGINE_CXXFLAGS) -MMD -MP -c $< -o $@
$(FLASH_V6_VERIFY_ORACLE): dev/benchmarks/flash_v6_verifier_attribution_oracle.mm \
        $(FLASH_V6_OBJECTS) $(FLASH_CORE_OBJECTS) $(LIB)
	$(RUN_CONFIGURED) $(CXX) -I$(FLASH_V6_SNAPSHOT) $(ENGINE_OBJCXXFLAGS) $< \
		$(FLASH_V6_OBJECTS) $(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@
.PHONY: flash-v6-verifier-attribution-oracle
flash-v6-verifier-attribution-oracle: $(FLASH_V6_VERIFY_ORACLE)
