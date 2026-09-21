# Private graph-execution experiment; public Forward ABI is unchanged.
SPLASH_PRECISION := hybrid
override ENGINE_CXXFLAGS := -Iruntime -Idev/benchmarks/prefill_frontier_split \
	-std=c++20 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
override ENGINE_OBJCXXFLAGS := $(ENGINE_CXXFLAGS) -fobjc-arc
include Makefile

$(FLASH_BUILD)/FlashForward.o: dev/benchmarks/prefill_frontier_split/FlashForward.cpp \
	dev/benchmarks/prefill_frontier_split/Diagnostic.hpp | $(FLASH_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_CXXFLAGS) -MMD -MP \
		-c dev/benchmarks/prefill_frontier_split/FlashForward.cpp -o $@

FRONTIER_ORACLE := $(BUILD)/prefill-frontier-split-oracle
$(FRONTIER_ORACLE): dev/benchmarks/prefill_frontier_split/oracle.mm \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(CXX) $(ENGINE_OBJCXXFLAGS) \
		dev/benchmarks/prefill_frontier_split/oracle.mm \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
		$(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@
