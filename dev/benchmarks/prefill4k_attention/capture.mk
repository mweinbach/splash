# Include after normal Makefile; only private capture overlays replace Forward.
PREFILL4K_ATTENTION_CAPTURE_BUILD ?= build/prefill4k-attention/capture
PREFILL4K_ATTENTION_CAPTURE_SOURCE := $(PREFILL4K_ATTENTION_CAPTURE_BUILD)/source/FlashForward.cpp
PREFILL4K_ATTENTION_CAPTURE_ATTRIBUTE := $(PREFILL4K_ATTENTION_CAPTURE_BUILD)/source/capture_attribution.mm
PREFILL4K_ATTENTION_CAPTURE_OBJECT := $(PREFILL4K_ATTENTION_CAPTURE_BUILD)/FlashForward.o
PREFILL4K_ATTENTION_CAPTURE := $(PREFILL4K_ATTENTION_CAPTURE_BUILD)/prefill4k-attention-capture
$(PREFILL4K_ATTENTION_CAPTURE_SOURCE): runtime/flash/FlashForward.cpp dev/benchmarks/prefill4k_attribution.mm dev/benchmarks/prefill4k_attention/generate_capture.py
	.venv/bin/python dev/benchmarks/prefill4k_attention/generate_capture.py $(PREFILL4K_ATTENTION_CAPTURE_BUILD)/source
$(PREFILL4K_ATTENTION_CAPTURE_ATTRIBUTE): $(PREFILL4K_ATTENTION_CAPTURE_SOURCE)
	test -f $@
$(PREFILL4K_ATTENTION_CAPTURE_OBJECT): $(PREFILL4K_ATTENTION_CAPTURE_SOURCE) dev/benchmarks/prefill4k_attention/capture.hpp
	$(CXX) $(ENGINE_CXXFLAGS) -Idev/benchmarks/prefill4k_attention -MMD -MP -c $< -o $@
$(PREFILL4K_ATTENTION_CAPTURE): dev/benchmarks/prefill4k_attention/capture_main.mm dev/benchmarks/prefill4k_attention/capture.hpp $(PREFILL4K_ATTENTION_CAPTURE_ATTRIBUTE) \
        $(PREFILL4K_ATTENTION_CAPTURE_OBJECT) $(filter-out $(FLASH_BUILD)/FlashForward.o $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(CXX) $(ENGINE_OBJCXXFLAGS) -Idev/benchmarks/prefill4k_attention -I$(PREFILL4K_ATTENTION_CAPTURE_BUILD)/source $< $(PREFILL4K_ATTENTION_CAPTURE_OBJECT) \
        $(filter-out $(FLASH_BUILD)/FlashForward.o $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@
.PHONY: prefill4k-attention-capture
prefill4k-attention-capture: $(PREFILL4K_ATTENTION_CAPTURE)
-include $(PREFILL4K_ATTENTION_CAPTURE_OBJECT:.o=.d)
