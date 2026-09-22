# Standalone CPU/shader component check; no GPU execution or payload hashing.
GATHERED_MPP_BUILD ?= build/prefill4k-allrows-gathered-mpp-component-v3
GATHERED_MPP_BASE ?= build/prefill4k-allrows-full512/source
GATHERED_MPP_SOURCE := $(GATHERED_MPP_BUILD)/source
GATHERED_MPP_MANIFEST := $(GATHERED_MPP_SOURCE)/gathered-mpp-manifest.json
$(GATHERED_MPP_MANIFEST): dev/benchmarks/prefill4k_allrows_gathered_mpp.py dev/benchmarks/prefill4k_allrows_gathered_mpp.hpp dev/benchmarks/prefill4k_allrows_gathered_mpp.metal
	.venv/bin/python dev/benchmarks/prefill4k_allrows_gathered_mpp.py --source $(GATHERED_MPP_BASE) --destination $(GATHERED_MPP_SOURCE)
$(GATHERED_MPP_BUILD)/gathered.air: $(GATHERED_MPP_MANIFEST)
	xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(GATHERED_MPP_SOURCE)/runtime -Iruntime -mmacosx-version-min=27.0 -c $(GATHERED_MPP_SOURCE)/runtime/metal/kernels/shared/flash_gathered_mpp.metal -o $@
$(GATHERED_MPP_BUILD)/gathered.metallib: $(GATHERED_MPP_BUILD)/gathered.air
	xcrun -sdk macosx metallib $< -o $@
$(GATHERED_MPP_BUILD)/Store.o: $(GATHERED_MPP_MANIFEST)
	xcrun -sdk macosx clang++ -std=c++20 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc -I$(GATHERED_MPP_SOURCE)/runtime -I$(GATHERED_MPP_BASE)/runtime -I$(GATHERED_MPP_BASE)/runtime/flash -Iruntime -c $(GATHERED_MPP_SOURCE)/runtime/flash/FlashInt8ExpertStore.mm -o $@
.PHONY: gathered-mpp-component-cpu
gathered-mpp-component-cpu: $(GATHERED_MPP_BUILD)/gathered.metallib $(GATHERED_MPP_BUILD)/Store.o
	.venv/bin/python dev/benchmarks/prefill4k_allrows_gathered_mpp.py --source $(GATHERED_MPP_BASE) --cpu-self-test
