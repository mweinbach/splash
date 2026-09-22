# Minimal private dense precision build. No target launches a backend/model/GPU.
DENSE_PRECISION_BUILD ?= build/prefill-dense-sep21
DENSE_BASELINE_BUILD ?= build/prefill4k-dense
DENSE_CXX := xcrun -sdk macosx clang++
DENSE_METAL := xcrun -sdk macosx metal
DENSE_FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0
DENSE_METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0
DENSE_BASE_AIRS := $(addprefix $(DENSE_BASELINE_BUILD)/,$(addsuffix .air,flash_dense_cache flash_dense_cache_m64 flash_dense_cache_prefill flash_dense_small_rows flash_affine candidate))

.PHONY: all cpu-self-test
all: $(DENSE_PRECISION_BUILD)/oracle $(DENSE_PRECISION_BUILD)/precision.metallib

$(DENSE_PRECISION_BUILD)/precision.air: $(DENSE_PRECISION_BUILD)/precision.metal runtime/metal/abi/FlashDenseCache.h runtime/metal/kernels/common/flash_dense_traversal.h
	$(DENSE_METAL) $(DENSE_METALFLAGS) -c $< -o $@
$(DENSE_PRECISION_BUILD)/register.air: dev/benchmarks/prefill_dense_sep21/register.metal runtime/metal/abi/FlashDenseCache.h runtime/metal/kernels/common/flash_dense_traversal.h
	$(DENSE_METAL) $(DENSE_METALFLAGS) -c $< -o $@
$(DENSE_PRECISION_BUILD)/precision.metallib: $(DENSE_PRECISION_BUILD)/precision.air $(DENSE_PRECISION_BUILD)/register.air $(DENSE_BASE_AIRS)
	xcrun -sdk macosx metallib $^ -o $@
$(DENSE_PRECISION_BUILD)/oracle.o: $(DENSE_PRECISION_BUILD)/oracle.mm
	$(DENSE_CXX) $(DENSE_FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(DENSE_PRECISION_BUILD)/oracle: $(DENSE_PRECISION_BUILD)/oracle.o $(DENSE_BASELINE_BUILD)/MetalBackend.o $(DENSE_BASELINE_BUILD)/DeviceCapabilities.o
	$(DENSE_CXX) $(DENSE_FLAGS) -fobjc-arc $^ -framework Foundation -framework Metal -framework IOKit -o $@
cpu-self-test: $(DENSE_PRECISION_BUILD)/oracle
	$(DENSE_PRECISION_BUILD)/oracle --cpu-self-test

-include $(DENSE_PRECISION_BUILD)/oracle.d
