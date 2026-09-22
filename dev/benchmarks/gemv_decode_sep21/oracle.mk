# Fresh CPU-build artifacts only. Nothing here invokes GPU mode or reads model
# payloads. Root explicitly runs --gpu after reviewing the frozen quality plan.
GEMV_ORACLE_BUILD ?= build/gemv-decode-sep21-one-layer-v1
GEMV_ORACLE_COMPONENT ?= build/prefill4k-allrows-gathered-mpp-component-v3
GEMV_ORACLE_BASE ?= build/prefill4k-allrows-full512
GEMV_ORACLE_C2 ?= build/prefill4k-allrows-qmv-c2-component
GEMV_ORACLE_OBJECTS := $(filter-out $(GEMV_ORACLE_BASE)/host/FlashWorker.o,$(wildcard $(GEMV_ORACLE_BASE)/host/*.o)) build/engine/metal/MetalBackend.o build/engine/metal/DeviceCapabilities.o build/engine/engine/Protocol.o build/engine/engine/MemoryGovernor.o
GEMV_ORACLE_AIRS := $(filter-out build/metal/shared/flash_int8_expert_store.air,$(wildcard build/metal/*/*.air)) $(GEMV_ORACLE_BASE)/metal/flash_int8_expert_store.air build/prefill4k-allrows-qmv-one-layer/probe.air
GEMV_ORACLE_FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -ffp-contract=off -fno-fast-math -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc -I$(GEMV_ORACLE_COMPONENT)/source/runtime -I$(GEMV_ORACLE_C2)/source/runtime -I$(GEMV_ORACLE_BASE)/source/runtime -Iruntime -Idev/benchmarks
$(GEMV_ORACLE_BUILD)/oracle: dev/benchmarks/gemv_decode_sep21/oracle.mm dev/benchmarks/gemv_decode_sep21/quality.hpp dev/benchmarks/gemv_decode_sep21/malformed.hpp dev/benchmarks/prefill4k_allrows_qmv_oracle.mm dev/benchmarks/prefill4k_allrows_qmv_reference.hpp dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp $(GEMV_ORACLE_OBJECTS)
	@mkdir -p $(dir $@)
	xcrun -sdk macosx clang++ $(GEMV_ORACLE_FLAGS) $< $(GEMV_ORACLE_OBJECTS) -framework Foundation -framework Metal -framework IOKit -o $@
$(GEMV_ORACLE_BUILD)/gemv-probe.air: dev/benchmarks/gemv_decode_sep21/probe.metal dev/benchmarks/gemv_decode_sep21/kernels.metal dev/benchmarks/prefill4k_allrows_gathered_mpp_probe.metal dev/benchmarks/prefill4k_allrows_gathered_mpp.metal dev/benchmarks/prefill4k_allrows_qmv_probe.h
	@mkdir -p $(dir $@)
	xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(GEMV_ORACLE_COMPONENT)/source/runtime -I$(GEMV_ORACLE_C2)/source/runtime -I$(GEMV_ORACLE_BASE)/source/runtime -Iruntime -Idev/benchmarks -mmacosx-version-min=27.0 -c $< -o $@
$(GEMV_ORACLE_BUILD)/splash.metallib: $(GEMV_ORACLE_BUILD)/gemv-probe.air $(GEMV_ORACLE_AIRS)
	xcrun -sdk macosx metallib $^ -o $@
.PHONY: gemv-decode-one-layer-cpu gemv-decode-one-layer-host-cpu
gemv-decode-one-layer-cpu: $(GEMV_ORACLE_BUILD)/oracle $(GEMV_ORACLE_BUILD)/splash.metallib
	$(GEMV_ORACLE_BUILD)/oracle --cpu-self-test
gemv-decode-one-layer-host-cpu: $(GEMV_ORACLE_BUILD)/oracle
	$(GEMV_ORACLE_BUILD)/oracle --cpu-self-test
