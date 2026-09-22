# Isolated compile target; no GPU work occurs while building.
SPLITK_BUILD ?= build/ultra-splitk
SPLITK_CXX := xcrun clang++
SPLITK_FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Iruntime -Idev/benchmarks -mmacosx-version-min=26.4
SPLITK_AIR := $(SPLITK_BUILD)/ultra_splitk.air
SPLITK_LIB := $(SPLITK_BUILD)/ultra_splitk.metallib
SPLITK_BACKEND := $(SPLITK_BUILD)/MetalBackend.o
SPLITK_CAPS := $(SPLITK_BUILD)/DeviceCapabilities.o
SPLITK_ORACLE := $(SPLITK_BUILD)/ultra-splitk-oracle
.PHONY: all cpu-self-test
all: $(SPLITK_ORACLE)
$(SPLITK_AIR): dev/benchmarks/ultra_splitk.metal dev/benchmarks/ultra_splitk_params.h
	mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal4.0 -O3 -Wall -Wextra -Werror -Idev/benchmarks -mmacosx-version-min=26.4 -c $< -o $@
$(SPLITK_LIB): $(SPLITK_AIR)
	xcrun -sdk macosx metallib $< -o $@
$(SPLITK_BACKEND): runtime/metal/MetalBackend.mm $(wildcard runtime/metal/*.hpp)
	mkdir -p $(@D)
	$(SPLITK_CXX) $(SPLITK_FLAGS) -fobjc-arc -c $< -o $@
$(SPLITK_CAPS): runtime/metal/DeviceCapabilities.cpp runtime/metal/DeviceCapabilities.hpp
	mkdir -p $(@D)
	$(SPLITK_CXX) $(SPLITK_FLAGS) -c $< -o $@
$(SPLITK_ORACLE): dev/benchmarks/ultra_splitk_oracle.mm dev/benchmarks/ultra_splitk_params.h \
    dev/benchmarks/FlashFloatBoundaryAudit.hpp $(SPLITK_BACKEND) $(SPLITK_CAPS) $(SPLITK_LIB)
	$(SPLITK_CXX) $(SPLITK_FLAGS) -fobjc-arc -ffp-contract=off $< \
	    $(SPLITK_BACKEND) $(SPLITK_CAPS) -framework Foundation -framework Metal -framework IOKit -o $@
cpu-self-test: $(SPLITK_ORACLE)
	$(SPLITK_ORACLE) --cpu-self-test
