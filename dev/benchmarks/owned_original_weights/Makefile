# Private original-byte owned Shared storage. No production ABI overlay.
SPLASH_PRECISION := hybrid
include Makefile

$(FLASH_BUILD)/FlashWeights.o: dev/benchmarks/owned_original_weights/FlashWeights.mm \
    dev/benchmarks/owned_original_weights/PrivateStorage.hpp | $(FLASH_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) -MMD -MP \
	    -c dev/benchmarks/owned_original_weights/FlashWeights.mm -o $@

$(FLASH_BUILD)/FlashWorker.o: dev/benchmarks/owned_original_weights/FlashWorker.mm \
    dev/benchmarks/owned_original_weights/PrivateStorage.hpp | $(FLASH_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) -MMD -MP \
	    -c dev/benchmarks/owned_original_weights/FlashWorker.mm -o $@
