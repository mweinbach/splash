PREFILL4K_GDN_CAPTURE_BUILD ?= build/prefill4k-gdn-capture
PREFILL4K_GDN_CAPTURE_SOURCE := $(PREFILL4K_GDN_CAPTURE_BUILD)/source/FlashForward.cpp
PREFILL4K_GDN_CAPTURE_OBJECT := $(PREFILL4K_GDN_CAPTURE_BUILD)/FlashForward.o
PREFILL4K_GDN_CAPTURE := $(PREFILL4K_GDN_CAPTURE_BUILD)/capture
$(PREFILL4K_GDN_CAPTURE_SOURCE): runtime/flash/FlashForward.cpp dev/benchmarks/prefill4k_dense/gdn_generate_capture.py
	.venv/bin/python dev/benchmarks/prefill4k_dense/gdn_generate_capture.py $@
$(PREFILL4K_GDN_CAPTURE_OBJECT): $(PREFILL4K_GDN_CAPTURE_SOURCE) dev/benchmarks/prefill4k_dense/gdn_capture.hpp
	$(CXX) $(ENGINE_CXXFLAGS) -Idev/benchmarks/prefill4k_dense -MMD -MP -c $< -o $@
$(PREFILL4K_GDN_CAPTURE): dev/benchmarks/prefill4k_dense/gdn_capture_main.mm dev/benchmarks/prefill4k_dense/gdn_capture.hpp \
        $(PREFILL4K_GDN_CAPTURE_OBJECT) $(filter-out $(FLASH_BUILD)/FlashForward.o $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(CXX) $(ENGINE_OBJCXXFLAGS) -Idev/benchmarks/prefill4k_dense $< $(PREFILL4K_GDN_CAPTURE_OBJECT) \
        $(filter-out $(FLASH_BUILD)/FlashForward.o $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
        $(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@
.PHONY: prefill4k-gdn-capture
prefill4k-gdn-capture: $(PREFILL4K_GDN_CAPTURE)
-include $(PREFILL4K_GDN_CAPTURE_OBJECT:.o=.d)
