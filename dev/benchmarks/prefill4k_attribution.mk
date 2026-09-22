# Unique instrumentation build; no production source edits or GPU execution.
PREFILL4K_ATTRIBUTION := $(BUILD)/prefill4k-attribution
$(PREFILL4K_ATTRIBUTION): BUILD_CONFIG := $(CONFIG_DIGEST)
$(PREFILL4K_ATTRIBUTION): dev/benchmarks/prefill4k_attribution.mm \
        $(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $< \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
		$(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@
.PHONY: prefill4k-attribution
prefill4k-attribution: $(PREFILL4K_ATTRIBUTION)

PREFILL4K_TEACHER_ORACLE := $(BUILD)/prefill4k-teacher-oracle
$(PREFILL4K_TEACHER_ORACLE): BUILD_CONFIG := $(CONFIG_DIGEST)
$(PREFILL4K_TEACHER_ORACLE): dev/benchmarks/prefill4k_attribution_teacher_oracle.mm dev/benchmarks/prefill4k_attribution.mm \
        $(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $< \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
		$(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@
.PHONY: prefill4k-teacher-oracle
prefill4k-teacher-oracle: $(PREFILL4K_TEACHER_ORACLE)
