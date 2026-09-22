# Standalone C1/C2 optional component build; compile + CPU checks only.
PREFILL4K_QMV_C2_BUILD ?= build/prefill4k-allrows-qmv-c2-component
PREFILL4K_QMV_C2_BASE ?= build/prefill4k-allrows-full512/source
PREFILL4K_QMV_C2_SOURCE := $(PREFILL4K_QMV_C2_BUILD)/source
PREFILL4K_QMV_C2_MANIFEST := $(PREFILL4K_QMV_C2_SOURCE)/qmv-overlay-manifest.json
PREFILL4K_QMV_C2_CPU := $(PREFILL4K_QMV_C2_BUILD)/qmv-c2-cpu
PREFILL4K_QMV_C2_AIR := $(PREFILL4K_QMV_C2_BUILD)/flash_gathered_i8_qmv_c2.air
PREFILL4K_QMV_C2_C1_AIR := $(PREFILL4K_QMV_C2_BUILD)/flash_gathered_i8_qmv.air
PREFILL4K_QMV_C2_LIB := $(PREFILL4K_QMV_C2_BUILD)/qmv-c1-c2.metallib
PREFILL4K_QMV_C2_STORE := $(PREFILL4K_QMV_C2_BUILD)/FlashInt8ExpertStore.o
$(PREFILL4K_QMV_C2_MANIFEST): dev/benchmarks/prefill4k_allrows_qmv.py dev/benchmarks/prefill4k_allrows_qmv.hpp dev/benchmarks/prefill4k_allrows_qmv.metal dev/benchmarks/prefill4k_allrows_qmv_c2.py dev/benchmarks/prefill4k_allrows_qmv_c2.metal
	$(PYTHON) dev/benchmarks/prefill4k_allrows_qmv_c2.py --source $(PREFILL4K_QMV_C2_BASE) --destination $(PREFILL4K_QMV_C2_SOURCE)
$(PREFILL4K_QMV_C2_CPU): dev/benchmarks/prefill4k_allrows_qmv_c2.cpp $(PREFILL4K_QMV_C2_MANIFEST)
	$(CXX) -I$(PREFILL4K_QMV_C2_SOURCE)/runtime $(ENGINE_CXXFLAGS) -ffp-contract=off -fno-fast-math $< -o $@
$(PREFILL4K_QMV_C2_AIR): $(PREFILL4K_QMV_C2_MANIFEST)
	$(METAL) $(PROD_METALFLAGS) -I$(PREFILL4K_QMV_C2_SOURCE)/runtime -c $(PREFILL4K_QMV_C2_SOURCE)/runtime/metal/kernels/shared/flash_gathered_i8_qmv_c2.metal -o $@
$(PREFILL4K_QMV_C2_C1_AIR): $(PREFILL4K_QMV_C2_MANIFEST)
	$(METAL) $(PROD_METALFLAGS) -I$(PREFILL4K_QMV_C2_SOURCE)/runtime -c $(PREFILL4K_QMV_C2_SOURCE)/runtime/metal/kernels/shared/flash_gathered_i8_qmv.metal -o $@
$(PREFILL4K_QMV_C2_LIB): $(PREFILL4K_QMV_C2_AIR) $(PREFILL4K_QMV_C2_C1_AIR)
	$(METALLIB) $^ -o $@
$(PREFILL4K_QMV_C2_STORE): $(PREFILL4K_QMV_C2_MANIFEST)
	$(CXX) -I$(PREFILL4K_QMV_C2_SOURCE)/runtime -I$(PREFILL4K_QMV_C2_BASE)/runtime -I$(PREFILL4K_QMV_C2_BASE)/runtime/flash $(ENGINE_OBJCXXFLAGS) -c $(PREFILL4K_QMV_C2_SOURCE)/runtime/flash/FlashInt8ExpertStore.mm -o $@
.PHONY: prefill4k-qmv-c2-component-cpu
prefill4k-qmv-c2-component-cpu: $(PREFILL4K_QMV_C2_CPU) $(PREFILL4K_QMV_C2_LIB) $(PREFILL4K_QMV_C2_STORE)
	$(PREFILL4K_QMV_C2_CPU) --cpu-self-test
	$(PYTHON) dev/benchmarks/prefill4k_allrows_qmv_c2.py --source $(PREFILL4K_QMV_C2_BASE) --cpu-self-test
