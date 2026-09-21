include Makefile

PLE_SSD_LOADER_ORACLE := $(BUILD)/flash-ple-ssd-loader-oracle
$(PLE_SSD_LOADER_ORACLE): BUILD_CONFIG := $(CONFIG_DIGEST)
$(PLE_SSD_LOADER_ORACLE): dev/benchmarks/flash_ple_ssd_loader_oracle.mm \
	$(FLASH_BUILD)/FlashWeights.o $(FLASH_BUILD)/FlashDescriptor.o \
	$(FLASH_BUILD)/FlashPLESSDStore.o $(ENGINE_BUILD)/metal/MetalBackend.o \
	$(ENGINE_BUILD)/metal/DeviceCapabilities.o
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $< $(FLASH_BUILD)/FlashWeights.o \
		$(FLASH_BUILD)/FlashDescriptor.o $(FLASH_BUILD)/FlashPLESSDStore.o \
		$(ENGINE_BUILD)/metal/MetalBackend.o $(ENGINE_BUILD)/metal/DeviceCapabilities.o \
		$(ENGINE_LINKFLAGS) -o $@
