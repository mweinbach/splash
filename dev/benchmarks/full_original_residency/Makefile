# Private Worker-only original residency union experiment; unchanged GPU graph.
SPLASH_PRECISION := hybrid
override ENGINE_CXXFLAGS := -Iruntime -Idev/benchmarks/full_original_residency \
	-std=c++20 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
override ENGINE_OBJCXXFLAGS := $(ENGINE_CXXFLAGS) -fobjc-arc
include Makefile

$(FLASH_BUILD)/FlashWorker.o: dev/benchmarks/full_original_residency/FlashWorker.mm \
	dev/benchmarks/full_original_residency/Policy.hpp | $(FLASH_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) -MMD -MP \
		-c dev/benchmarks/full_original_residency/FlashWorker.mm -o $@
