# Private original-head instrumentation; production sources/math are unmodified.
FLASH_TRAINED_HEAD_PROFILE_ORACLE := $(BUILD)/flash-trained-head-profile-v8-oracle
$(FLASH_TRAINED_HEAD_PROFILE_ORACLE): BUILD_CONFIG := $(CONFIG_DIGEST)
$(FLASH_TRAINED_HEAD_PROFILE_ORACLE): dev/benchmarks/flash_trained_head_profile_v8_oracle.mm \
        $(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $< \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
		$(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@
.PHONY: flash-trained-head-profile-v8-oracle
flash-trained-head-profile-v8-oracle: $(FLASH_TRAINED_HEAD_PROFILE_ORACLE)
