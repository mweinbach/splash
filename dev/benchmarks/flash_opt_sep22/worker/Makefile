# Optimized Flash-Next worker (base: R5-integer-currentQ4-fixed4-sep22-worker-v2).
#   make -j12            -> splash-flash + splash.metallib in this directory
#   make install DEST=build/flash-opt-sep22   (copies the runtime pair)
W := $(abspath .)
SRC := $(W)/source
CXX := xcrun -sdk macosx clang++
METAL := xcrun -sdk macosx metal
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -I$(SRC)/runtime -I$(SRC) -I$(SRC)/dev/benchmarks/prefill4k_attention
METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -I$(SRC)/runtime -mmacosx-version-min=27.0
include sources.mk
HOST := $(addprefix host/,$(addsuffix .o,$(REBUILD_NAMES)))
CORE := core/047-MetalBackend.o core/048-DeviceCapabilities.o core/049-Protocol.o core/050-MemoryGovernor.o
# Prebuilt AIR inputs of the base library, in its original link order, then
# the kernels added by this work.
EXTRA_AIRS := $(patsubst kernels/%.metal,air/extra-%.air,$(wildcard kernels/*.metal))
AIRS := $(shell cat air-order.txt) $(EXTRA_AIRS)
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
.PHONY: all install clean
all: splash-flash splash.metallib
define host_rule
host/$(1).o: $$(SRC_$(1))
	@mkdir -p host
	$$(CXX) $$(FLAGS) -MMD -MP -c $$< -o $$@
endef
$(foreach n,$(REBUILD_NAMES),$(eval $(call host_rule,$(n))))
core/047-MetalBackend.o: $(SRC)/runtime/metal/MetalBackend.mm
	@mkdir -p core
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
core/048-DeviceCapabilities.o: $(SRC)/runtime/metal/DeviceCapabilities.cpp
	@mkdir -p core
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
core/049-Protocol.o: $(SRC)/runtime/engine/Protocol.cpp
	@mkdir -p core
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
core/050-MemoryGovernor.o: $(SRC)/runtime/engine/MemoryGovernor.cpp
	@mkdir -p core
	$(CXX) $(FLAGS) -MMD -MP -c $< -o $@
air/extra-%.air: kernels/%.metal $(wildcard kernels/*.h)
	$(METAL) $(METALFLAGS) -c $< -o $@
splash-flash: $(HOST) $(CORE)
	$(CXX) $(FLAGS) $^ $(FRAMEWORKS) -o $@
splash.metallib: $(AIRS)
	xcrun -sdk macosx metallib $^ -o $@
install: splash-flash splash.metallib
	@test -n "$(DEST)" || (echo "DEST=<directory> required" && exit 1)
	mkdir -p $(DEST)
	cp splash-flash splash.metallib $(DEST)/
clean:
	rm -rf host core splash-flash splash.metallib air/extra-*.air
-include $(HOST:.o=.d) $(CORE:.o=.d)
