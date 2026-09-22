PROFILE_BUILD ?= build/mtp-teacher-bulk-profile-sep22-v1
BUILD := build/mtp-teacher-bulk-sep21-v4
include $(BUILD)/link-inputs.mk
HOST := $(addprefix $(BUILD)/host/,$(addsuffix .o,$(REBUILD_NAMES)))
CXX := xcrun -sdk macosx clang++
FLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -fobjc-arc -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 -I$(PROFILE_BUILD)/source -I$(PROFILE_BUILD)/source/runtime -I$(PROFILE_BUILD)/source/dev/benchmarks/prefill4k_attention
.PHONY: all cpu-self-test
all: $(PROFILE_BUILD)/teacher-bulk-profile
$(PROFILE_BUILD)/teacher-bulk-profile: $(PROFILE_BUILD)/source/dev/benchmarks/mtp_teacher_bulk_sep21/profile.mm $(BUILD)/host/teacher_bulk.o $(HOST) $(REUSED) $(CORE)
	$(CXX) $(FLAGS) -MMD -MP $< $(BUILD)/host/teacher_bulk.o $(HOST) $(REUSED) $(CORE) -framework Foundation -framework Metal -framework IOKit -o $@
cpu-self-test: all
	$(PROFILE_BUILD)/teacher-bulk-profile --cpu-self-test
	$(PROFILE_BUILD)/teacher-bulk-profile --help
-include $(PROFILE_BUILD)/teacher-bulk-profile.d
