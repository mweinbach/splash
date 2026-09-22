# Optional standalone CPU/shader check. Never loads a model or creates a GPU.
PREFILL4K_QMV_BUILD ?= build/prefill4k-allrows-qmv-component
PREFILL4K_QMV_BASE ?= build/prefill4k-allrows-full512/source
PREFILL4K_QMV_SOURCE := $(PREFILL4K_QMV_BUILD)/source
PREFILL4K_QMV_MANIFEST := $(PREFILL4K_QMV_SOURCE)/qmv-overlay-manifest.json
PREFILL4K_QMV_CPU := $(PREFILL4K_QMV_BUILD)/qmv-cpu
PREFILL4K_QMV_AIR := $(PREFILL4K_QMV_BUILD)/flash_gathered_i8_qmv.air
PREFILL4K_QMV_LIB := $(PREFILL4K_QMV_BUILD)/qmv.metallib
PREFILL4K_QMV_STORE := $(PREFILL4K_QMV_BUILD)/FlashInt8ExpertStore.o
$(PREFILL4K_QMV_MANIFEST): dev/benchmarks/prefill4k_allrows_qmv.py dev/benchmarks/prefill4k_allrows_qmv.hpp dev/benchmarks/prefill4k_allrows_qmv.metal
	$(PYTHON) dev/benchmarks/prefill4k_allrows_qmv.py --source $(PREFILL4K_QMV_BASE) --destination $(PREFILL4K_QMV_SOURCE)
$(PREFILL4K_QMV_CPU): dev/benchmarks/prefill4k_allrows_qmv.cpp dev/benchmarks/prefill4k_allrows_qmv.hpp
	@mkdir -p $(dir $@)
	$(CXX) $(ENGINE_CXXFLAGS) -Idev/benchmarks $< -o $@
$(PREFILL4K_QMV_AIR): $(PREFILL4K_QMV_MANIFEST)
	$(METAL) $(PROD_METALFLAGS) -I$(PREFILL4K_QMV_SOURCE)/runtime -c $(PREFILL4K_QMV_SOURCE)/runtime/metal/kernels/shared/flash_gathered_i8_qmv.metal -o $@
$(PREFILL4K_QMV_LIB): $(PREFILL4K_QMV_AIR)
	$(METALLIB) $< -o $@
$(PREFILL4K_QMV_STORE): $(PREFILL4K_QMV_MANIFEST)
	$(CXX) -I$(PREFILL4K_QMV_SOURCE)/runtime -I$(PREFILL4K_QMV_BASE)/runtime -I$(PREFILL4K_QMV_BASE)/runtime/flash $(ENGINE_OBJCXXFLAGS) -c $(PREFILL4K_QMV_SOURCE)/runtime/flash/FlashInt8ExpertStore.mm -o $@
.PHONY: prefill4k-qmv-component-cpu
prefill4k-qmv-component-cpu: $(PREFILL4K_QMV_CPU) $(PREFILL4K_QMV_LIB) $(PREFILL4K_QMV_STORE)
	$(PREFILL4K_QMV_CPU) --cpu-self-test
	$(PYTHON) dev/benchmarks/prefill4k_allrows_qmv.py --source $(PREFILL4K_QMV_BASE) --cpu-self-test
