# Private indirect compute dispatch

This is a private backend/header overlay. Production `runtime/metal` files are
unchanged. `ComputeDispatch` ABI differs, so every host operator and CommandGraph
user must rebuild against the overlay; linking v6 operator objects is unsafe.

Build through the private wrapper in a fresh directory:

```sh
make -f dev/benchmarks/indirect_dispatch_backend/Makefile \
  BUILD=build/private-indirect-runtime -j8 flash-next
```

The wrapper sets overlay flags before normal build configuration identity is
computed and compiles the cloned backend source. Private CommandGraph,
DeviceQueries, and ProfilingJson headers prevent quote includes from resolving
to the production sibling header.

Register a mutable argument buffer once, with every immutable model operand
that it must not alias:

```cpp
auto source = backend.registerIndirectDispatchSource(arguments, immutableOperands);
auto dispatch = originalDispatch;
dispatch.indirectGroups = ComputeDispatch::IndirectGroups{source, slotByteOffset};
```

The registered owner token retains its buffer. Registration rejects an empty or
foreign backend allocation, sparse/argument buffers, unaligned base views,
less than 12 bytes, and overlaps into supplied immutable operands. Submission
requires a registered source from the submitting backend plus a four-byte
aligned offset and complete 12-byte extent. Ordinary pipeline, buffer/bytes,
duplicate-index, thread-product, and pipeline thread-limit checks remain active.

The GPU owns three uint32 group dimensions per slot, including zero dimensions.
Encoding declares the source as Read, inserts a resource barrier after prior
producer writes, and calls Metal's indirect threadgroup dispatch API. The host
never reads those dimensions. Command tickets retain the source allocation even
when caller graphs/tokens/buffers are destroyed. No additional backing is
charged. Normal command timing stays in the original submission path. Private
profiling marks groups as indirect and reports the slot offset; original static
groups describe maxima rather than actual GPU-produced dimensions.

Root-only primitive command:

```sh
build/private-indirect-dispatch-primitive-v1/oracle \
  build/private-indirect-dispatch-primitive-v1/splash.metallib
```

The oracle uses a Private argument buffer, GPU-produced active/zero-axis slots,
exact guarded copy checks, host rejection cases, and indirect-only ticket
retention after caller destruction. It performs no dimension host readback.
Its six CPU self-checks and the 165 standalone range/thread/overlap checks
passed; the standalone policy also passed ASan/UBSan. Root then ran the private
GPU primitive successfully: six commands, four active/zero-axis scenarios,
zero-dispatch no writes, exact copy/canaries, ticket retention, and zero host
dimension readbacks. This qualifies the primitive needed for EOS-safe chain
testing; it does not establish full-model/service correctness or performance.
