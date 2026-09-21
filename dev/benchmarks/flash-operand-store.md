# Persisted affine operands

The local operand store saves the already qualified dense BF16 and F32 cache
coefficients in separate, read-only files. Loading a matching file can replace
the cache's startup allocation and GPU reconstruction. It preserves the source
checkpoint and does not establish an inference speed gain by itself. Changes to
kernel arithmetic or precision still require their own numerical and HTTP
qualification.

The `splash-local-affine-operands-v1` manifest binds the artifact to both the
source identity and the original aligned weight manifest fingerprint. It also
binds the coefficient math version: separate F32 multiply and add with
contraction disabled, followed by round-to-nearest-even when storing BF16.
Each source view includes its dimensions, affine bits/group, and packed weight
and parameter strides. An operand for another source view cannot be reused.

Each entry occupies a unique flat ASCII `.bin` filename. Its offset is zero,
its logical extent is exactly `N*K*sizeof(format)`, and its allocated extent
rounds up to 16 KiB with zero padding. The payload SHA256 covers the entire
stored allocation, including padding. The separate `manifest.sha256` covers
the exact manifest bytes and contains one lowercase digest plus a newline.
Only BF16 and F32 dense operands are accepted; this artifact does not introduce
INT8 activation arithmetic.

Build the exporter in a private build directory, then run checkpoint conversion
as a single GPU job after stopping inference services:

```sh
make BUILD=build/flash-operand-export build/flash-operand-export/flash-operand-export
build/flash-operand-export/flash-operand-export \
  build/flash-operand-export/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  install/local-models/Flash-Next-operands-v1
```

The default selection saves every BF16 dense-cache operand, including the
vocabulary head, and every F32 body operand. F32 vocabulary storage is excluded
because the qualified original-code INT8 head no longer needs it. Use
`--include-f32-head`, `--bf16-only`, `--f32-only`, or explicit projection prefixes
after the three paths to change the selection. A destination must be fresh and
outside the source package and original `~/.omlx` directory. Publication uses
a private staging directory, synchronized files, and an exclusive atomic rename;
abandoned staging files are removed without changing an existing artifact.

`SPLASH_FLASH_OPERAND_STORE=install/local-models/Flash-Next-operands-v1` makes the
two cache constructors validate this store against the current source, then
SHA256-verify and map each selected file with `PROT_READ`. Unstored projections
use their existing conversion path. A present entry with a wrong source,
format, geometry, math version, size, or payload checksum rejects construction.
The backend charges each mapped allocation once; the cache's existing
`plannedBytes` estimate remains conservative, including conversion diagnostics.
`persistedTensorCount()` and `operandStoreIdentitySha256()` report stored usage
without changing the numerical cache identity.

The stdlib fixture tool creates two tiny signed-affine operands (32 KiB total)
and verifies the manifest and actual payload bytes without importing MLX or
creating a Metal device:

```sh
.venv/bin/python -m dev.tests.flash.flash_operand_store_fixtures /tmp/flash-operand-fixture --create
.venv/bin/python -m unittest dev.tests.flash.test_flash_operand_store
```

When the native exporter has been built, the same tests can exercise its
production CPU-only parser and payload verification:

```sh
FLASH_OPERAND_STORE_NATIVE_CHECKER=build/flash-operand-store/flash-operand-export \
  .venv/bin/python -m unittest dev.tests.flash.test_flash_operand_store
```

The cases cover exact byte hashes, source and math identity, duplicate JSON
keys/operands/files, strict integers, quantization geometry and strides,
traversal/symlink escapes, bounds, truncation/growth, allocation alignment, and
padding corruption with a recomputed checksum. A metadata-only check explicitly
does not claim payload integrity; selected operands must be payload-verified
before mapping for inference.

The private native checker passed all 19 tests on September 20, 2026 in 1.416
seconds. This includes native production parsing and byte verification for both
operand formats. The native CPU writer/loader self-test also passed publication,
existing-destination rejection, abandoned staging cleanup, and source identity
gating. These checks instantiate no Metal device. Full checkpoint conversion,
mapped-cache numerical checks, and service throughput remain separate GPU jobs.

## Read-only mapping ownership audit

The September 20 source audit found no gap in the mapping owner chain. This is
source evidence; the CPU parser tests do not exercise Metal buffer creation or
asynchronous completion.

`Mapping::open` creates a `PROT_READ`, `MAP_SHARED` mapping and closes its file
descriptor after `mmap`. `Mapping::~Mapping` is the only unmap operation.
`FlashOperandStore::mapTensor` gives the mapping's shared owner to
`MetalBackend::wrapSharedMemory`; the returned tensor does not depend on the
store object's lifetime. The cache constructors retain each tensor while their
local store object is destroyed at the end of construction.

The backend keeps that owner in two places. `MetalAllocation::externalOwner`
owns it through all C++ buffer references, and Metal's buffer deallocator
captures another shared owner through any Metal buffer references. The explicit
C++ owner also covers validation wrappers that fail to retain the deallocator
block. Because `externalOwner` is the allocation's first declared member, it is
destroyed after the allocation's strong Metal buffer member.

| Scenario | Source evidence |
|---|---|
| Store destroyed after mapping | `runtime/flash/FlashOperandStore.mm:230` owns the mapping; `:416` transfers the owner to the returned tensor's buffer. |
| Tensor or buffer copied | `runtime/metal/MetalBackend.hpp:70` declares shared buffer ownership; `runtime/metal/MetalBackend.mm:975` defaults copy/move operations. |
| Buffer view retained | `runtime/metal/MetalBackend.mm:1670` shares the original `MetalAllocation` with the view. |
| Graph retained after cache destruction | `runtime/metal/CommandGraph.hpp:56` stores owning buffer bindings in each dispatch. |
| Graph/cache destroyed after asynchronous submission | `runtime/metal/MetalBackend.mm:2146` copies every bound allocation into the ticket before submission, including indirect source allocations. |
| Ticket waited on, replaced, or destroyed | `runtime/metal/MetalBackend.mm:665` retains allocations; `:1052` waits for completion before releasing an abandoned ticket, and `:1084` releases only after a completed wait. |
| Metal retains a buffer after ticket completion | `runtime/metal/MetalBackend.mm:1625` captures the mapping owner in the buffer deallocator; `:196` independently owns it in the C++ allocation. |
| Cache moved or destroyed | `runtime/flash/FlashDenseCache.cpp:192` and `runtime/flash/FlashFloatDenseCache.cpp:338` move/destroy the owning implementation; tensor copies, views, graph bindings, and tickets remain independent owners. |

A borrowed `const FlashTensor&` from a cache still follows normal C++ reference
lifetime rules: copy its buffer/tensor or create an owning backend view before
destroying the cache. Moving a cache transfers its implementation without
changing mapped allocations; replacing an existing cache destroys the previous
implementation.

Dynamic GPU qualification should cover both formats by mapping a tensor,
destroying the original store, preserving a copy/view in a graph, submitting an
asynchronous read, then destroying the graph and cache before waiting for the
result. Matching outputs establish runtime behavior. Checking reclamation must
allow Metal's completion handler and autorelease owners to release the buffer;
ticket completion alone does not promise an immediate `munmap`.
