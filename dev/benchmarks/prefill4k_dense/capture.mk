# Include after normal Makefile, reusing unchanged normal objects/library.
PREFILL4K_DENSE_CAPTURE_BUILD ?= build/prefill4k-dense/capture
PREFILL4K_DENSE_CAPTURE_SOURCE := $(PREFILL4K_DENSE_CAPTURE_BUILD)/source/FlashForward.cpp
PREFILL4K_DENSE_CAPTURE_OBJECT := $(PREFILL4K_DENSE_CAPTURE_BUILD)/FlashForward.o
PREFILL4K_DENSE_CAPTURE := $(PREFILL4K_DENSE_CAPTURE_BUILD)/prefill4k-dense-capture
$(PREFILL4K_DENSE_CAPTURE_SOURCE): runtime/flash/FlashForward.cpp dev/benchmarks/prefill4k_dense/generate_capture.py
	.venv/bin/python dev/benchmarks/prefill4k_dense/generate_capture.py $@
$(PREFILL4K_DENSE_CAPTURE_OBJECT): $(PREFILL4K_DENSE_CAPTURE_SOURCE) dev/benchmarks/prefill4k_dense/capture.hpp
	$(CXX) $(ENGINE_CXXFLAGS) -Idev/benchmarks/prefill4k_dense -MMD -MP -c $< -o $@
$(PREFILL4K_DENSE_CAPTURE): dev/benchmarks/prefill4k_dense/capture_main.mm dev/benchmarks/prefill4k_dense/capture.hpp \
        $(PREFILL4K_DENSE_CAPTURE_OBJECT) $(filter-out $(FLASH_BUILD)/FlashForward.o $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(CXX) $(ENGINE_OBJCXXFLAGS) -Idev/benchmarks/prefill4k_dense $< $(PREFILL4K_DENSE_CAPTURE_OBJECT) \
        $(filter-out $(FLASH_BUILD)/FlashForward.o $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
        $(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@
.PHONY: prefill4k-dense-capture
prefill4k-dense-capture: $(PREFILL4K_DENSE_CAPTURE)
-include $(PREFILL4K_DENSE_CAPTURE_OBJECT:.o=.d)
