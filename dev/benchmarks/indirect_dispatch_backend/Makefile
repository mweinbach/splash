# Private wrapper: set host overlay flags before normal config identity is
# computed. Always pass a fresh BUILD directory on the make command line.
SPLASH_PRECISION := hybrid
override ENGINE_CXXFLAGS := -Idev/benchmarks/indirect_dispatch_backend/include \
	-Idev/benchmarks/indirect_dispatch_backend/include/metal -Iruntime -Iruntime/metal \
	-std=c++20 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1
override ENGINE_OBJCXXFLAGS := $(ENGINE_CXXFLAGS) -fobjc-arc
include Makefile
include dev/benchmarks/indirect_dispatch_backend/private.mk
