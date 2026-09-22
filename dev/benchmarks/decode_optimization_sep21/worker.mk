# Include with the normal Makefile, after generating the fresh private tree.
PREFILL4K_WIDE_BUILD ?= build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1
PREFILL4K_WIDE_FULLCACHE := 1
PREFILL4K_WIDE_EXTRA_SHADER_NAMES := flash_gathered_mpp
include dev/benchmarks/prefill4k_wide.mk
include dev/benchmarks/prefill4k_attention/bulk_runtime.mk

DECODE_SEP21_POLICY_CPU := $(PREFILL4K_WIDE_BUILD)/row-cap-policy-cpu
$(DECODE_SEP21_POLICY_CPU): dev/benchmarks/decode_optimization_sep21/policy_cpu.cpp $(PREFILL4K_WIDE_MANIFEST)
	$(CXX) $(PREFILL4K_WIDE_FLAGS) $(ENGINE_CXXFLAGS) $< -o $@
.PHONY: decode-sep21-worker-cpu
decode-sep21-worker-cpu: $(PREFILL4K_WIDE_WORKER) $(DECODE_SEP21_POLICY_CPU)
	$(PREFILL4K_WIDE_WORKER) --cpu-self-test
	$(DECODE_SEP21_POLICY_CPU)
