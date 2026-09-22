# CPU/compile-only preparation; GPU execution stays with root's exclusive slot.
ATTENTION_BF16P_BUILD ?= build/prefill-attention-bf16p-sep21-v1
ATTENTION_BASE ?= build/prefill4k-attention
ATTENTION_SOURCE := dev/benchmarks/prefill4k_attention
ATTENTION_FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -Iruntime -I$(ATTENTION_SOURCE) -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
ATTENTION_METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0
ATTENTION_OBJECTS := $(addprefix $(ATTENTION_BASE)/,MetalBackend.o DeviceCapabilities.o FlashQSA.o FlashQSAFast.o FlashQSAMPP.o coalesced.o bulk.o)
ATTENTION_AIRS := $(addprefix $(ATTENTION_BASE)/,flash_qsa.air flash_qsa_fast.air flash_qsa_mpp.air candidate.air bulk_attention_sg8.air bulk_dense_direct.air bulk_dense_direct_bf16pv.air)

.PHONY: all cpu-self-test
all: $(ATTENTION_BF16P_BUILD)/oracle $(ATTENTION_BF16P_BUILD)/candidate.metallib

$(ATTENTION_BF16P_BUILD)/candidate.air: $(ATTENTION_BF16P_BUILD)/candidate.metal
	xcrun -sdk macosx metal $(ATTENTION_METALFLAGS) -c $< -o $@
$(ATTENTION_BF16P_BUILD)/candidate.metallib: $(ATTENTION_BF16P_BUILD)/candidate.air $(ATTENTION_AIRS)
	xcrun -sdk macosx metallib $^ -o $@
$(ATTENTION_BF16P_BUILD)/oracle.o: $(ATTENTION_BF16P_BUILD)/oracle.mm $(ATTENTION_SOURCE)/oracle.mm $(ATTENTION_SOURCE)/bulk.hpp
	xcrun -sdk macosx clang++ $(ATTENTION_FLAGS) -fobjc-arc -MMD -MP -c $< -o $@
$(ATTENTION_BF16P_BUILD)/oracle: $(ATTENTION_BF16P_BUILD)/oracle.o $(ATTENTION_OBJECTS)
	xcrun -sdk macosx clang++ $(ATTENTION_FLAGS) -fobjc-arc $^ -framework Foundation -framework Metal -framework IOKit -o $@
cpu-self-test: $(ATTENTION_BF16P_BUILD)/oracle
	$(ATTENTION_BF16P_BUILD)/oracle --cpu-self-test

-include $(ATTENTION_BF16P_BUILD)/oracle.d
