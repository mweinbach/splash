PREFILL4K_WIDE_BUILD ?= build/prefill4k-qsa-bulk-gathered-bf16-sep21-v2
include dev/benchmarks/decode_optimization_sep21/worker.mk
DECODE_SEP21_BF16_CPU := $(PREFILL4K_WIDE_BUILD)/bf16-policy-cpu
$(DECODE_SEP21_BF16_CPU): dev/benchmarks/decode_optimization_sep21/bf16_policy_cpu.cpp $(PREFILL4K_WIDE_MANIFEST)
	$(CXX) $(PREFILL4K_WIDE_FLAGS) $(ENGINE_CXXFLAGS) $< -o $@
.PHONY: decode-sep21-bf16-worker-cpu
decode-sep21-bf16-worker-cpu: decode-sep21-worker-cpu $(DECODE_SEP21_BF16_CPU)
	$(DECODE_SEP21_BF16_CPU)
