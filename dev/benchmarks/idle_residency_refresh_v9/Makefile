# Private backend-only idle residency refresh control; public ABI unchanged.
SPLASH_PRECISION := hybrid
override ENGINE_CXXFLAGS := -Iruntime -Idev/benchmarks/idle_residency_refresh_v9 \
	-std=c++20 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
override ENGINE_OBJCXXFLAGS := $(ENGINE_CXXFLAGS) -fobjc-arc
include Makefile

$(ENGINE_BUILD)/metal/MetalBackend.o: dev/benchmarks/idle_residency_refresh_v9/MetalBackend.mm \
	dev/benchmarks/idle_residency_refresh_v9/Policy.hpp | $(ENGINE_BUILD)
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) -MMD -MP \
		-c dev/benchmarks/idle_residency_refresh_v9/MetalBackend.mm -o $@

IDLE_ORACLE := $(BUILD)/idle-residency-refresh-v9-oracle
$(IDLE_ORACLE): dev/benchmarks/idle_residency_refresh_v9/oracle.mm \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(CXX) $(ENGINE_OBJCXXFLAGS) \
		dev/benchmarks/idle_residency_refresh_v9/oracle.mm \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
		$(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@
