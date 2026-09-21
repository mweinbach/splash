# CPU policy and scheduling review

The private policy reference passed 7,941 checks under AddressSanitizer and
UndefinedBehaviorSanitizer. No GPU work was executed by this reviewer, and no
production file was edited.

The test covers all 512 combinations of eligibility signals, strict 0/1 opt-in,
all 901 canonical intervals from 100 through 1000 milliseconds and invalid
spellings, host reserve boundaries including UINT64_MAX overflow, every model
geometry field, and individual corruption of all 4096 output words. It rejects
an owner-word permutation even though the XOR checksum is unchanged. The
checksum supplements exact source-word and guard validation.

The Worker must require no active requests, no pending requests, no live IDs,
no in-flight request command, no outstanding backend command or sparse unmap,
an empty reader queue, healthy execution, and no shutdown. An active request
waiting for a grammar mask is not idle. `!worked` alone is insufficient.

Host policy must use a fresh governor snapshot after updating the asynchronous
system-pressure signal. Both effective and system pressure must be normal,
host measurement valid, growth allowed, reservations zero, and available
memory strictly greater than the reserve plus 2 GiB. This maintenance only
references already-accounted owners; it does not add their 144.33 GB to the
allocation ledger again. The host estimate credits pageable file-backed pages,
so this margin is a suspension policy rather than a physical-wiring promise.

Startup and constructor commands must not arm maintenance. The current private
Worker arms only after a real request emitted tokens and completed normally,
which is conservative. Maintenance cannot rearm itself.

The final incoming-queue check should follow status publication and precede
command submission as closely as possible. Holding the transport queue mutex
across GPU work would impede control processing and is inappropriate. A reader
arrival after commitment can overlap one synchronous maintenance command.
The 500 ms interval is not a latency bound: after a cold miss, pressure
suspension, or delayed host scheduling, that one command may pay the entire
driver preparation delay. Record actual maximum wall duration and cold misses;
consider disarming after a cold miss until a real request succeeds again.

Maintenance counts and timings must remain separate from request prefill,
decode, MTP acceptance and token accounting. A maintenance error must clear the
maintenance in-flight marker during teardown; it must not publish a successful
completion or corrupt mutable inference buffers.

CPU reproduction:

```sh
xcrun clang++ -std=c++20 -Wall -Wextra -Werror -O1 -g \
  -fsanitize=address,undefined \
  dev/benchmarks/idle_residency_maintenance_v11/policy_test.cpp \
  -o build/private-idle-maintenance-policy-v11/policy_test
build/private-idle-maintenance-policy-v11/policy_test
```

Result: `build/private-idle-maintenance-policy-v11/policy-qualification.json`.
