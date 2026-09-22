# Bounded ONE-layer primitive source build. CPU target never executes GPU mode.
QMV_ORACLE_BUILD ?= build/prefill4k-allrows-qmv-one-layer
QMV_ORACLE_BASE ?= build/prefill4k-allrows-full512
QMV_ORACLE_C2 ?= build/prefill4k-allrows-qmv-c2-component
QMV_ORACLE_HOST := $(filter-out $(QMV_ORACLE_BASE)/host/FlashWorker.o,$(wildcard $(QMV_ORACLE_BASE)/host/*.o))
QMV_ORACLE_CORE := build/engine/metal/MetalBackend.o build/engine/metal/DeviceCapabilities.o build/engine/engine/Protocol.o build/engine/engine/MemoryGovernor.o
QMV_ORACLE_AIRS := $(filter-out build/metal/shared/flash_int8_expert_store.air,$(wildcard build/metal/*/*.air)) $(QMV_ORACLE_BASE)/metal/flash_int8_expert_store.air
QMV_ORACLE_FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -ffp-contract=off -fno-fast-math -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc -I$(QMV_ORACLE_C2)/source/runtime -I$(QMV_ORACLE_BASE)/source/runtime -Iruntime -Idev/benchmarks
$(QMV_ORACLE_BUILD)/oracle: dev/benchmarks/prefill4k_allrows_qmv_oracle.mm dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp dev/benchmarks/prefill4k_allrows_qmv_probe.h dev/benchmarks/prefill4k_allrows_qmv_reference.hpp $(QMV_ORACLE_HOST) $(QMV_ORACLE_CORE)
	@mkdir -p $(dir $@)
	xcrun -sdk macosx clang++ $(QMV_ORACLE_FLAGS) $< $(QMV_ORACLE_HOST) $(QMV_ORACLE_CORE) -framework Foundation -framework Metal -framework IOKit -o $@
$(QMV_ORACLE_BUILD)/probe.air: dev/benchmarks/prefill4k_allrows_qmv_probe.metal dev/benchmarks/prefill4k_allrows_qmv_probe.h
	@mkdir -p $(dir $@)
	xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -Idev/benchmarks -mmacosx-version-min=27.0 -c $< -o $@
$(QMV_ORACLE_BUILD)/splash.metallib: $(QMV_ORACLE_BUILD)/probe.air $(QMV_ORACLE_C2)/flash_gathered_i8_qmv.air $(QMV_ORACLE_C2)/flash_gathered_i8_qmv_c2.air $(QMV_ORACLE_AIRS)
	xcrun -sdk macosx metallib $^ -o $@
.PHONY: qmv-one-layer-cpu
qmv-one-layer-cpu: $(QMV_ORACLE_BUILD)/oracle $(QMV_ORACLE_BUILD)/splash.metallib
	$(QMV_ORACLE_BUILD)/oracle --cpu-self-test
