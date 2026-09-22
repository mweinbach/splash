# CPU-build only; Root explicitly invokes --gpu after the frozen parity plan.
GATHERED_ORACLE_BUILD ?= build/prefill4k-allrows-gathered-stage-one-layer-sep21-v2a
GATHERED_ORACLE_COMPONENT ?= build/prefill4k-allrows-gathered-mpp-component-v3
GATHERED_ORACLE_BASE ?= build/prefill4k-allrows-full512
GATHERED_ORACLE_OBJECTS := $(filter-out $(GATHERED_ORACLE_BASE)/host/FlashWorker.o,$(wildcard $(GATHERED_ORACLE_BASE)/host/*.o)) build/engine/metal/MetalBackend.o build/engine/metal/DeviceCapabilities.o build/engine/engine/Protocol.o build/engine/engine/MemoryGovernor.o
GATHERED_ORACLE_AIRS := $(filter-out build/metal/shared/flash_int8_expert_store.air,$(wildcard build/metal/*/*.air)) $(GATHERED_ORACLE_BASE)/metal/flash_int8_expert_store.air build/prefill4k-allrows-qmv-one-layer/probe.air
GATHERED_ORACLE_FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -ffp-contract=off -fno-fast-math -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc -I$(GATHERED_ORACLE_COMPONENT)/source/runtime -Ibuild/prefill4k-allrows-qmv-c2-component/source/runtime -I$(GATHERED_ORACLE_BASE)/source/runtime -Iruntime -Idev/benchmarks
$(GATHERED_ORACLE_BUILD)/oracle: dev/benchmarks/gathered_stage_sep21_v2a/oracle.mm dev/benchmarks/prefill4k_allrows_qmv_oracle.mm dev/benchmarks/prefill4k_allrows_gathered_mpp.hpp
	@mkdir -p $(dir $@)
	xcrun -sdk macosx clang++ $(GATHERED_ORACLE_FLAGS) $< $(GATHERED_ORACLE_OBJECTS) -framework Foundation -framework Metal -framework IOKit -o $@
$(GATHERED_ORACLE_BUILD)/gathered-probe.air: dev/benchmarks/gathered_stage_sep21_v2a/probe.metal dev/benchmarks/prefill4k_allrows_gathered_stage_probe.metal dev/benchmarks/prefill4k_allrows_gathered_stage.metal dev/benchmarks/prefill4k_allrows_gathered_mpp_probe.metal dev/benchmarks/prefill4k_allrows_gathered_mpp.metal dev/benchmarks/prefill4k_allrows_qmv_probe.h dev/benchmarks/gathered_stage_sep21_v2a/gathered_sg1_probe.metal dev/benchmarks/gathered_stage_sep21_v2a/gathered_sg2_probe.metal dev/benchmarks/prefill_moe_sep21/gathered_sg1.metal dev/benchmarks/prefill_moe_sep21/gathered_sg2.metal
	@mkdir -p $(dir $@)
	xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror -I$(GATHERED_ORACLE_COMPONENT)/source/runtime -Iruntime -Idev/benchmarks -mmacosx-version-min=27.0 -c $< -o $@
$(GATHERED_ORACLE_BUILD)/splash.metallib: $(GATHERED_ORACLE_BUILD)/gathered-probe.air $(GATHERED_ORACLE_AIRS)
	xcrun -sdk macosx metallib $^ -o $@
.PHONY: gathered-stage-one-layer-cpu
gathered-stage-one-layer-cpu: $(GATHERED_ORACLE_BUILD)/oracle $(GATHERED_ORACLE_BUILD)/splash.metallib
	$(GATHERED_ORACLE_BUILD)/oracle --cpu-self-test
