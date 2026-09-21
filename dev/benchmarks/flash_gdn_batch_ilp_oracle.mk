# Private compile-only build. Explicit execution is a separate Root action.
GDN_BATCH_BUILD ?= build/flash-gdn-batch-ilp-v1
GDN_BATCH_BASELINE_BUILD ?= build/flash-default-v5
GDN_BATCH_CXX := xcrun -sdk macosx clang++
GDN_BATCH_FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -Iruntime -I$(GDN_BATCH_BUILD) -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
GDN_BATCH_CPP := runtime/flash/FlashGDN.cpp runtime/flash/FlashGDNFused.cpp runtime/flash/FlashGDNSeparate.cpp runtime/flash/FlashGDNStaged.cpp
GDN_BATCH_MM := runtime/metal/MetalBackend.mm
GDN_BATCH_CPP_OBJ := $(patsubst runtime/flash/%.cpp,$(GDN_BATCH_BUILD)/fresh-%.o,$(GDN_BATCH_CPP))
GDN_BATCH_MM_OBJ := $(patsubst runtime/metal/%.mm,$(GDN_BATCH_BUILD)/fresh-%.o,$(GDN_BATCH_MM))
GDN_BATCH_OBJECTS := $(GDN_BATCH_CPP_OBJ) $(GDN_BATCH_MM_OBJ) $(GDN_BATCH_BUILD)/fresh-DeviceCapabilities.o $(GDN_BATCH_BUILD)/flash_gdn_batch_ilp.o
GDN_BATCH_AIRS := $(shell rg --files $(GDN_BATCH_BASELINE_BUILD)/metal -g '*.air')
GDN_BATCH_HEADERS := $(shell rg --files runtime/metal runtime/flash -g '*.hpp' -g '*.h')

.PHONY: all
all: $(GDN_BATCH_BUILD)/flash-gdn-batch-ilp-oracle $(GDN_BATCH_BUILD)/splash.metallib

$(GDN_BATCH_BUILD):
	mkdir -p $@

$(GDN_BATCH_BUILD)/fresh-%.o: runtime/flash/%.cpp $(GDN_BATCH_HEADERS) | $(GDN_BATCH_BUILD)
	$(GDN_BATCH_CXX) $(GDN_BATCH_FLAGS) -c $< -o $@

$(GDN_BATCH_BUILD)/fresh-%.o: runtime/metal/%.mm $(GDN_BATCH_HEADERS) | $(GDN_BATCH_BUILD)
	$(GDN_BATCH_CXX) $(GDN_BATCH_FLAGS) -fobjc-arc -c $< -o $@

$(GDN_BATCH_BUILD)/fresh-DeviceCapabilities.o: runtime/metal/DeviceCapabilities.cpp $(GDN_BATCH_HEADERS) | $(GDN_BATCH_BUILD)
	$(GDN_BATCH_CXX) $(GDN_BATCH_FLAGS) -c $< -o $@

$(GDN_BATCH_BUILD)/flash_gdn_batch_ilp.o: dev/benchmarks/flash_gdn_batch_ilp.cpp dev/benchmarks/flash_gdn_batch_ilp.hpp dev/benchmarks/flash_gdn_batch_ilp.h $(GDN_BATCH_HEADERS) | $(GDN_BATCH_BUILD)
	$(GDN_BATCH_CXX) $(GDN_BATCH_FLAGS) -c $< -o $@

$(GDN_BATCH_BUILD)/flash_gdn_batch_ilp.air: dev/benchmarks/flash_gdn_batch_ilp.metal dev/benchmarks/flash_gdn_batch_ilp.h | $(GDN_BATCH_BUILD)
	xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -c $< -o $@

$(GDN_BATCH_BUILD)/BatchGdnOracleBuildProvenance.hpp: $(GDN_BATCH_OBJECTS) dev/benchmarks/flash_gdn_batch_ilp_oracle.mm dev/tests/engine/flash_gdn_metal_test.mm dev/benchmarks/flash_gdn_batch_ilp_oracle.mk
	.venv/bin/python -c 'import hashlib,json,pathlib,sys; paths=[pathlib.Path(p) for p in sys.argv[2:]]; d={"build":"fresh-abi-private-batched-gdn-ilp-v1","all_linked_runtime_objects_recompiled_here":True,"command_timing_size_expected":200,"files":[{"path":str(p.resolve()),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size} for p in paths]}; target=pathlib.Path(sys.argv[1]); encoded=json.dumps(d,separators=(",",":")); target.write_text("#pragma once\ninline constexpr const char *kBatchGdnOracleBuildProvenance = R\"JSON("+encoded+")JSON\";\n"); target.with_suffix(".json").write_text(encoded+"\n")' $@ $(GDN_BATCH_OBJECTS) $(GDN_BATCH_CPP) $(GDN_BATCH_MM) runtime/metal/DeviceCapabilities.cpp runtime/metal/MetalBackend.hpp dev/benchmarks/flash_gdn_batch_ilp_oracle.mm dev/tests/engine/flash_gdn_metal_test.mm dev/benchmarks/flash_gdn_batch_ilp.cpp dev/benchmarks/flash_gdn_batch_ilp.hpp dev/benchmarks/flash_gdn_batch_ilp.h dev/benchmarks/flash_gdn_batch_ilp.metal dev/benchmarks/flash_gdn_batch_ilp_oracle.mk

$(GDN_BATCH_BUILD)/flash-gdn-batch-ilp-oracle: dev/benchmarks/flash_gdn_batch_ilp_oracle.mm dev/tests/engine/flash_gdn_metal_test.mm $(GDN_BATCH_BUILD)/BatchGdnOracleBuildProvenance.hpp $(GDN_BATCH_OBJECTS)
	$(GDN_BATCH_CXX) $(GDN_BATCH_FLAGS) -fobjc-arc $< $(GDN_BATCH_OBJECTS) -framework Foundation -framework Metal -framework IOKit -o $@

$(GDN_BATCH_BUILD)/splash.metallib: $(GDN_BATCH_AIRS) $(GDN_BATCH_BUILD)/flash_gdn_batch_ilp.air
	test -n '$(GDN_BATCH_AIRS)'
	xcrun -sdk macosx metallib $(GDN_BATCH_AIRS) $(GDN_BATCH_BUILD)/flash_gdn_batch_ilp.air -o $@
