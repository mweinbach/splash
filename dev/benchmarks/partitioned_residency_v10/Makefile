# Private partitioned residency diagnostic. Rebuild all host objects against the private overlay.
SPLASH_PRECISION := hybrid
override ENGINE_CXXFLAGS := -Idev/benchmarks/partitioned_residency_v10 -Idev/benchmarks/partitioned_residency_v10/metal -Iruntime -Iruntime/metal \
	-std=c++20 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
override ENGINE_OBJCXXFLAGS := $(ENGINE_CXXFLAGS) -fobjc-arc
include Makefile

$(ENGINE_BUILD)/metal/MetalBackend.o: dev/benchmarks/partitioned_residency_v10/MetalBackend.mm \
	dev/benchmarks/partitioned_residency_v10/Policy.hpp dev/benchmarks/partitioned_residency_v10/metal/MetalBackend.hpp | $(ENGINE_BUILD)
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) -Iruntime/metal -MMD -MP \
		-c dev/benchmarks/partitioned_residency_v10/MetalBackend.mm -o $@

$(FLASH_BUILD)/FlashWorker.o: dev/benchmarks/partitioned_residency_v10/FlashWorker.mm \
	dev/benchmarks/partitioned_residency_v10/Policy.hpp dev/benchmarks/full_original_residency/Policy.hpp | $(FLASH_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) -Iruntime/flash -MMD -MP \
		-c dev/benchmarks/partitioned_residency_v10/FlashWorker.mm -o $@
