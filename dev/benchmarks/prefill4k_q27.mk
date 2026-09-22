# Private compilation only. Root owns GPU execution.
Q27_BUILD ?= build/prefill4k-q27
Q27_CXX := xcrun -sdk macosx clang++
Q27_CXXFLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Iruntime -Idev/benchmarks -mmacosx-version-min=27.0
Q27_BACKEND := $(Q27_BUILD)/MetalBackend.o
Q27_CAPS := $(Q27_BUILD)/DeviceCapabilities.o
Q27_AIR := $(Q27_BUILD)/prefill4k_q27.air
Q27_LIB := $(Q27_BUILD)/prefill4k_q27.metallib
Q27_ORACLE := $(Q27_BUILD)/prefill4k-q27-oracle
.PHONY: all cpu-self-test
all: $(Q27_ORACLE)
$(Q27_AIR): dev/benchmarks/prefill4k_q27.metal dev/benchmarks/prefill4k_q27_params.h \
    dev/benchmarks/prefill4k_q27_q4_direct.h \
    dev/benchmarks/prefill4k_q27_q4_small.h \
    dev/benchmarks/prefill4k_q27_q4_pair.h \
    runtime/metal/kernels/prefill/linear_q4.metal $(wildcard runtime/metal/kernels/common/*.h) \
    $(wildcard runtime/metal/abi/*.h)
	mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -Idev/benchmarks -mmacosx-version-min=27.0 -c $< -o $@
$(Q27_LIB): $(Q27_AIR)
	xcrun -sdk macosx metallib $< -o $@
$(Q27_BACKEND): runtime/metal/MetalBackend.mm $(wildcard runtime/metal/*.hpp)
	mkdir -p $(@D)
	$(Q27_CXX) $(Q27_CXXFLAGS) -fobjc-arc -c $< -o $@
$(Q27_CAPS): runtime/metal/DeviceCapabilities.cpp runtime/metal/DeviceCapabilities.hpp
	mkdir -p $(@D)
	$(Q27_CXX) $(Q27_CXXFLAGS) -c $< -o $@
$(Q27_ORACLE): dev/benchmarks/prefill4k_q27_oracle.mm dev/benchmarks/prefill4k_q27_params.h \
    dev/benchmarks/FlashFloatBoundaryAudit.hpp $(Q27_BACKEND) $(Q27_CAPS) $(Q27_LIB)
	$(Q27_CXX) $(Q27_CXXFLAGS) -fobjc-arc -ffp-contract=off $< $(Q27_BACKEND) $(Q27_CAPS) \
	    -framework Foundation -framework Metal -framework IOKit -o $@
cpu-self-test: $(Q27_ORACLE)
	$(Q27_ORACLE) --cpu-self-test
