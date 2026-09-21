# Isolated host and shader compilation. No recipe executes a GPU/model.
GDN_LAZY_BUILD ?= build/flash-gdn-lazy-rollback-v1
GDN_LAZY_BASELINE_BUILD ?= build/flash-default-v6
GDN_LAZY_CXX := xcrun -sdk macosx clang++
GDN_LAZY_FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -Iruntime -I$(GDN_LAZY_BUILD) -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
GDN_LAZY_CPP := runtime/flash/FlashGDN.cpp runtime/flash/FlashGDNFused.cpp runtime/flash/FlashGDNSeparate.cpp
GDN_LAZY_CPP_OBJECTS := $(patsubst runtime/flash/%.cpp,$(GDN_LAZY_BUILD)/fresh-%.o,$(GDN_LAZY_CPP))
GDN_LAZY_OBJECTS := $(GDN_LAZY_CPP_OBJECTS) $(GDN_LAZY_BUILD)/fresh-MetalBackend.o $(GDN_LAZY_BUILD)/fresh-DeviceCapabilities.o $(GDN_LAZY_BUILD)/flash_gdn_lazy_rollback.o
GDN_LAZY_HEADERS := $(shell rg --files runtime/metal runtime/flash -g '*.hpp' -g '*.h')
GDN_LAZY_AIRS := $(shell rg --files $(GDN_LAZY_BASELINE_BUILD)/metal -g '*.air')

.PHONY: all
all: $(GDN_LAZY_BUILD)/flash-gdn-lazy-rollback-oracle $(GDN_LAZY_BUILD)/splash.metallib

$(GDN_LAZY_BUILD):
	mkdir -p $@

$(GDN_LAZY_BUILD)/fresh-%.o: runtime/flash/%.cpp $(GDN_LAZY_HEADERS) | $(GDN_LAZY_BUILD)
	$(GDN_LAZY_CXX) $(GDN_LAZY_FLAGS) -c $< -o $@

$(GDN_LAZY_BUILD)/fresh-MetalBackend.o: runtime/metal/MetalBackend.mm $(GDN_LAZY_HEADERS) | $(GDN_LAZY_BUILD)
	$(GDN_LAZY_CXX) $(GDN_LAZY_FLAGS) -fobjc-arc -c $< -o $@

$(GDN_LAZY_BUILD)/fresh-DeviceCapabilities.o: runtime/metal/DeviceCapabilities.cpp $(GDN_LAZY_HEADERS) | $(GDN_LAZY_BUILD)
	$(GDN_LAZY_CXX) $(GDN_LAZY_FLAGS) -c $< -o $@

$(GDN_LAZY_BUILD)/flash_gdn_lazy_rollback.o: dev/benchmarks/flash_gdn_lazy_rollback.cpp dev/benchmarks/flash_gdn_lazy_rollback.hpp dev/benchmarks/flash_gdn_lazy_rollback.h $(GDN_LAZY_HEADERS) | $(GDN_LAZY_BUILD)
	$(GDN_LAZY_CXX) $(GDN_LAZY_FLAGS) -c $< -o $@

$(GDN_LAZY_BUILD)/flash_gdn_lazy_rollback.air: dev/benchmarks/flash_gdn_lazy_rollback.metal dev/benchmarks/flash_gdn_lazy_rollback.h | $(GDN_LAZY_BUILD)
	xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -c $< -o $@

$(GDN_LAZY_BUILD)/LazyGdnOracleBuildProvenance.hpp: $(GDN_LAZY_OBJECTS) dev/benchmarks/flash_gdn_lazy_rollback_oracle.mm dev/tests/engine/flash_gdn_metal_test.mm dev/benchmarks/flash_gdn_lazy_rollback_oracle.mk
	.venv/bin/python -c 'import hashlib,json,pathlib,sys; paths=[pathlib.Path(p) for p in sys.argv[2:]]; d={"build":"fresh-default-v6-abi-private-lazy-gdn-rollback-v1","all_linked_runtime_objects_recompiled_here":True,"command_timing_size_expected":200,"files":[{"path":str(p.resolve()),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size} for p in paths]}; target=pathlib.Path(sys.argv[1]); encoded=json.dumps(d,separators=(",",":")); target.write_text("#pragma once\ninline constexpr const char *kLazyGdnOracleBuildProvenance = R\"JSON("+encoded+")JSON\";\n"); target.with_suffix(".json").write_text(encoded+"\n")' $@ $(GDN_LAZY_OBJECTS) $(GDN_LAZY_CPP) runtime/metal/MetalBackend.mm runtime/metal/DeviceCapabilities.cpp runtime/metal/MetalBackend.hpp dev/benchmarks/flash_gdn_lazy_rollback_oracle.mm dev/tests/engine/flash_gdn_metal_test.mm dev/benchmarks/flash_gdn_lazy_rollback.cpp dev/benchmarks/flash_gdn_lazy_rollback.hpp dev/benchmarks/flash_gdn_lazy_rollback.h dev/benchmarks/flash_gdn_lazy_rollback.metal dev/benchmarks/flash_gdn_lazy_rollback_oracle.mk

$(GDN_LAZY_BUILD)/flash-gdn-lazy-rollback-oracle: dev/benchmarks/flash_gdn_lazy_rollback_oracle.mm dev/tests/engine/flash_gdn_metal_test.mm $(GDN_LAZY_BUILD)/LazyGdnOracleBuildProvenance.hpp $(GDN_LAZY_OBJECTS)
	$(GDN_LAZY_CXX) $(GDN_LAZY_FLAGS) -fobjc-arc $< $(GDN_LAZY_OBJECTS) -framework Foundation -framework Metal -framework IOKit -o $@

$(GDN_LAZY_BUILD)/splash.metallib: $(GDN_LAZY_AIRS) $(GDN_LAZY_BUILD)/flash_gdn_lazy_rollback.air
	test -n '$(GDN_LAZY_AIRS)'
	xcrun -sdk macosx metallib $(GDN_LAZY_AIRS) $(GDN_LAZY_BUILD)/flash_gdn_lazy_rollback.air -o $@
