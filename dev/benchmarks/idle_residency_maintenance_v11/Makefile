# Private default-off idle-maintenance Worker overlay; production is untouched.
SPLASH_PRECISION := hybrid
include Makefile

$(FLASH_BUILD)/FlashWorker.o: dev/benchmarks/idle_residency_maintenance_v11/FlashWorker.mm \
    dev/benchmarks/idle_residency_maintenance_v11/Maintenance.hpp \
    dev/benchmarks/idle_residency_maintenance_v11/Scheduler.hpp \
    dev/benchmarks/idle_residency_maintenance_v11/Policy.hpp | $(FLASH_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) -Iruntime/flash -MMD -MP \
		-c dev/benchmarks/idle_residency_maintenance_v11/FlashWorker.mm -o $@

$(METAL_BUILD)/private_idle_maintenance_v11.air: dev/benchmarks/idle_residency_maintenance_v11/maintenance.metal | $(METAL_BUILD)
	$(RUN_CONFIGURED) $(METAL) $(PROD_METALFLAGS) -c $< -o $@

$(LIB): $(METAL_BUILD)/private_idle_maintenance_v11.air
