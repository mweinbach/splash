# Isolated benchmark; include after the project Makefile.
# make -f Makefile -f dev/benchmarks/flash_qsa_mpp_oracle.mk \
#   BUILD=build/flash-qsa-mpp build/flash-qsa-mpp/flash-qsa-mpp-oracle
FLASH_QSA_MPP_ORACLE := $(BUILD)/flash-qsa-mpp-oracle
FLASH_QSA_MPP_ORACLE_OBJECTS := $(FLASH_BUILD)/FlashQSAMPP.o \
	$(FLASH_BUILD)/FlashQSAFast.o $(FLASH_BUILD)/FlashQSA.o $(FLASH_CORE_OBJECTS)

$(FLASH_QSA_MPP_ORACLE) $(FLASH_BUILD)/FlashQSAMPP.o: BUILD_CONFIG := $(CONFIG_DIGEST)

$(FLASH_QSA_MPP_ORACLE): dev/benchmarks/flash_qsa_mpp_oracle.mm \
		runtime/flash/FlashQSAMPP.hpp runtime/flash/FlashQSAFast.hpp \
		$(FLASH_QSA_MPP_ORACLE_OBJECTS) $(LIB)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $< \
		$(FLASH_QSA_MPP_ORACLE_OBJECTS) $(ENGINE_LINKFLAGS) -o $@

.PHONY: flash-qsa-mpp-oracle
flash-qsa-mpp-oracle: $(FLASH_QSA_MPP_ORACLE)
