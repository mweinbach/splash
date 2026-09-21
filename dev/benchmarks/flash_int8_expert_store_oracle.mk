# Private compile/link only; the GPU screen is run explicitly by Root.
# The frozen runtime dependency objects must already exist.
STORE_ORACLE_BUILD ?= build/flash-expert-int8/production-oracle
STORE_ORACLE_RUNTIME ?= build/flash-int8-expert-runtime
STORE_ORACLE_CXX ?= xcrun clang++
STORE_ORACLE_FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Iruntime \
	-mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -fobjc-arc
STORE_ORACLE_FLASH := $(filter-out $(STORE_ORACLE_RUNTIME)/flash/FlashWorker.o,\
	$(wildcard $(STORE_ORACLE_RUNTIME)/flash/*.o))
STORE_ORACLE_CORE := $(STORE_ORACLE_RUNTIME)/engine/metal/MetalBackend.o \
	$(STORE_ORACLE_RUNTIME)/engine/metal/DeviceCapabilities.o \
	$(STORE_ORACLE_RUNTIME)/engine/engine/Protocol.o \
	$(STORE_ORACLE_RUNTIME)/engine/engine/MemoryGovernor.o
STORE_ORACLE_BINARY := $(STORE_ORACLE_BUILD)/flash-int8-expert-store-oracle

.PHONY: all cpu-self-test
all: $(STORE_ORACLE_BINARY) $(STORE_ORACLE_BUILD)/splash.metallib

$(STORE_ORACLE_BUILD):
	mkdir -p $@

$(STORE_ORACLE_BUILD)/flash_int8_expert_store_oracle.o: \
	dev/benchmarks/flash_int8_expert_store_oracle.mm \
	dev/benchmarks/flash_expert_int8_bucket_reference.hpp \
	runtime/flash/FlashInt8ExpertStore.hpp \
	runtime/flash/FlashInt8ExpertStoreMetadata.hpp | $(STORE_ORACLE_BUILD)
	$(STORE_ORACLE_CXX) $(STORE_ORACLE_FLAGS) -c $< -o $@

$(STORE_ORACLE_BINARY): $(STORE_ORACLE_BUILD)/flash_int8_expert_store_oracle.o \
	$(STORE_ORACLE_FLASH) $(STORE_ORACLE_CORE)
	$(STORE_ORACLE_CXX) $(STORE_ORACLE_FLAGS) $^ \
		-framework Foundation -framework Metal -framework IOKit -o $@

$(STORE_ORACLE_BUILD)/splash.metallib: $(STORE_ORACLE_RUNTIME)/splash.metallib | $(STORE_ORACLE_BUILD)
	cp $< $@

cpu-self-test: all
	$(STORE_ORACLE_BINARY) --cpu-self-test
