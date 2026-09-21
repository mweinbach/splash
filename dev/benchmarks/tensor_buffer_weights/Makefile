# Private original-weight loader experiment; fresh ordinary v6 host ABI.
SPLASH_PRECISION := hybrid
override ENGINE_CXXFLAGS := -ivfsoverlay dev/benchmarks/tensor_buffer_weights/vfs-overlay.json -Iruntime \
	-std=c++20 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
override ENGINE_OBJCXXFLAGS := $(ENGINE_CXXFLAGS) -fobjc-arc
include Makefile

$(FLASH_BUILD)/FlashWeights.o: dev/benchmarks/tensor_buffer_weights/FlashWeights.mm \
	dev/benchmarks/tensor_buffer_weights/include/flash/FlashWeights.hpp
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) -MMD -MP \
		-c dev/benchmarks/tensor_buffer_weights/FlashWeights.mm -o $@

$(FLASH_BUILD)/FlashWorker.o: dev/benchmarks/tensor_buffer_weights/FlashWorker.mm \
	dev/benchmarks/tensor_buffer_weights/include/flash/FlashWeights.hpp
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) -MMD -MP \
		-c dev/benchmarks/tensor_buffer_weights/FlashWorker.mm -o $@
