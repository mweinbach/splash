# Private backend-only completed command retention; public ABI unchanged.
SPLASH_PRECISION := hybrid
override ENGINE_CXXFLAGS := -Iruntime -Iruntime/metal -Idev/benchmarks/completed_command_retention_v9 \
  -std=c++20 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
override ENGINE_OBJCXXFLAGS := $(ENGINE_CXXFLAGS) -fobjc-arc
include Makefile
$(ENGINE_BUILD)/metal/MetalBackend.o: dev/benchmarks/completed_command_retention_v9/MetalBackend.mm \
  dev/benchmarks/completed_command_retention_v9/Policy.hpp | $(ENGINE_BUILD)
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) -MMD -MP \
	  -c dev/benchmarks/completed_command_retention_v9/MetalBackend.mm -o $@
