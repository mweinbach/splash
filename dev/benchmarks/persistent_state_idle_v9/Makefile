# Private oracle-only native state identity diagnostic; production ABI unchanged.
SPLASH_PRECISION := hybrid
override ENGINE_CXXFLAGS := -Iruntime -Idev/benchmarks/persistent_state_idle_v9 \
	-std=c++20 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
override ENGINE_OBJCXXFLAGS := $(ENGINE_CXXFLAGS) -fobjc-arc
include Makefile
STATE_IDLE_ORACLE := $(BUILD)/persistent-state-idle-v9-oracle
$(STATE_IDLE_ORACLE): dev/benchmarks/persistent_state_idle_v9/oracle.mm \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) $(FLASH_CORE_OBJECTS) $(LIB)
	$(CXX) $(ENGINE_OBJCXXFLAGS) \
		dev/benchmarks/persistent_state_idle_v9/oracle.mm \
		$(filter-out $(FLASH_BUILD)/FlashWorker.o,$(FLASH_OBJECTS)) \
		$(FLASH_CORE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@
