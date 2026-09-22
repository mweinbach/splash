# Compile diagnostics against the already built immutable private worker objects.
DECODE_SEP21_PRIVATE_BUILD ?= build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1
DECODE_SEP21_ATTRIBUTION_BUILD ?= build/sep21-gathered-verifier-attribution-v1
DECODE_SEP21_ATTRIBUTION := $(DECODE_SEP21_ATTRIBUTION_BUILD)/verify-attribution
DECODE_SEP21_PRIVATE_OBJECTS := $(filter-out $(DECODE_SEP21_PRIVATE_BUILD)/host/FlashWorker.o,$(wildcard $(DECODE_SEP21_PRIVATE_BUILD)/host/*.o))
$(DECODE_SEP21_ATTRIBUTION): dev/benchmarks/decode_optimization_sep21/verify_attribution.mm $(DECODE_SEP21_PRIVATE_OBJECTS) $(FLASH_CORE_OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) -I$(DECODE_SEP21_PRIVATE_BUILD)/source/runtime -I$(DECODE_SEP21_PRIVATE_BUILD)/source/dev/benchmarks/prefill4k_attention $(ENGINE_OBJCXXFLAGS) -Wno-deprecated-declarations $< $(DECODE_SEP21_PRIVATE_OBJECTS) $(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@
.PHONY: decode-sep21-verifier-attribution-cpu
decode-sep21-verifier-attribution-cpu: $(DECODE_SEP21_ATTRIBUTION)
	$(DECODE_SEP21_ATTRIBUTION) --cpu-self-test
