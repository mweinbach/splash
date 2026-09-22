# Isolated ABI build: every host object uses the private indirect header.
# make -f dev/benchmarks/ultra_draft_chain.mk \
#   BUILD=build/ultra-draft-chain -j8 ultra-draft-chain-oracle
include dev/benchmarks/indirect_dispatch_backend/Makefile

ULTRA_DRAFT_CHAIN_ORACLE := $(BUILD)/ultra-draft-chain-oracle
ULTRA_DRAFT_CHAIN_LIB := $(BUILD)/ultra-draft-chain.metallib
ULTRA_DRAFT_CHAIN_GUARD := $(BUILD)/ultra-draft-chain-guard.air
ULTRA_DRAFT_CHAIN_HEAD := $(BUILD)/ultra-draft-chain-head.o
ULTRA_DRAFT_CHAIN_WORKSPACE := $(BUILD)/ultra-draft-chain-workspace.o
ULTRA_DRAFT_CHAIN_OBJECTS := $(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
	$(FLASH_CORE_OBJECTS) $(ULTRA_DRAFT_CHAIN_HEAD) $(ULTRA_DRAFT_CHAIN_WORKSPACE)

$(ULTRA_DRAFT_CHAIN_ORACLE) $(ULTRA_DRAFT_CHAIN_LIB) \
$(ULTRA_DRAFT_CHAIN_GUARD) $(ULTRA_DRAFT_CHAIN_HEAD) \
$(ULTRA_DRAFT_CHAIN_WORKSPACE): BUILD_CONFIG := $(CONFIG_DIGEST)

$(ULTRA_DRAFT_CHAIN_HEAD): dev/benchmarks/FlashMTPGPUChainFourCandidate.cpp \
		dev/benchmarks/FlashMTPGPUChainFourCandidate.hpp \
		dev/benchmarks/FlashMTPGPUChainFourCandidateState.hpp | $(BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_CXXFLAGS) -Idev/benchmarks -MMD -MP -c $< -o $@

$(ULTRA_DRAFT_CHAIN_WORKSPACE): dev/benchmarks/ultra_draft_chain_workspace.cpp \
		dev/benchmarks/ultra_draft_chain_workspace.hpp \
		dev/benchmarks/FlashMTPGPUChainFourGuard.h | $(BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_CXXFLAGS) -Idev/benchmarks -MMD -MP -c $< -o $@

$(ULTRA_DRAFT_CHAIN_GUARD): dev/benchmarks/flash_mtp_gpu_chain_four_guard.metal \
		dev/benchmarks/FlashMTPGPUChainFourGuard.h | $(BUILD)
	$(RUN_CONFIGURED) $(METAL) $(PROD_METALFLAGS) -Idev/benchmarks -c $< -o $@

$(ULTRA_DRAFT_CHAIN_LIB): $(PRODUCTION_AIRS) $(ULTRA_DRAFT_CHAIN_GUARD)
	$(RUN_CONFIGURED) $(METALLIB) $(BUILD_INPUTS) -o $@

$(ULTRA_DRAFT_CHAIN_ORACLE): dev/benchmarks/ultra_draft_chain_oracle.mm \
		dev/benchmarks/ultra_draft_chain_workspace.hpp \
		$(ULTRA_DRAFT_CHAIN_OBJECTS) $(ULTRA_DRAFT_CHAIN_LIB)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) -Idev/benchmarks $< \
		$(ULTRA_DRAFT_CHAIN_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

-include $(ULTRA_DRAFT_CHAIN_HEAD:.o=.d) $(ULTRA_DRAFT_CHAIN_WORKSPACE:.o=.d)

.PHONY: ultra-draft-chain-oracle
ultra-draft-chain-oracle: $(ULTRA_DRAFT_CHAIN_ORACLE)
