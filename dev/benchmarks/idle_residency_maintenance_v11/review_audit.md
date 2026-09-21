# Independent maintenance review

This review ran no GPU work and edited only `review_test.cpp` and this note.
The private Worker overlay keeps inference kernels, token selection, cookies,
and mutable model state unchanged. It schedules the readonly owner-touch
command only after a normally completed request with output, at a safe point
with no active, pending, live, incoming, or in-flight requests, no outstanding
backend command or unmap, and no shutdown. A fresh governor snapshot requires
normal system and effective pressure, valid host telemetry, zero reservations,
and strictly more than the host reserve plus 2 GiB available.

The selected sources are existing original immutable base allocations plus
verified derived operands, checked against the qualified source/layout and
the exact union of 1,134 owners and 144,326,852,608 bytes. The shader reads one
32-bit word from each readonly pointer and writes only its own guarded output.
Argument-buffer reflection validates readonly access, exact pointer count,
pointer alignment and extents. Existing native owners remain retained; their
weight bytes are not charged to the allocation ledger again. Only the small
argument buffer and 16 KiB diagnostics are newly allocated and governed.

Issues found during implementation and corrected by the owner include explicit
CPU pointer alignment, overlapping source rejection, conservative native
allocation planning with actual ledger reporting, an incoming-control check
after status publication, healthy-backend maintenance failure fallback, broken
trace sink removal, and disarming maintenance after a cold command until
another successful user request.

The independent AddressSanitizer/UndefinedBehaviorSanitizer runs passed:

- 7,941 policy checks: 512 eligibility combinations, all 901 allowed intervals,
  reserve/overflow boundaries, qualified geometry, and corruption of all 4,096
  output words.
- 37,842 checks of the actual `Scheduler.hpp` called by the Worker: startup,
  GPU-command-only and cancelled/zero-output completion do not arm; exact timer
  boundaries; user GPU reset; pressure/maintenance retry; cold miss disarm;
  maintenance cannot rearm itself; successful user rearm; failure and shutdown
  guards.

Evidence is in `build/private-idle-maintenance-review-v11/`.

The interval is not a wall-latency bound. An arrival after final commitment can
overlap one synchronous maintenance command, and a pressure suspension or host
scheduling gap may make that command pay the full driver rewiring delay.
Maximum wall time and cold-miss counters must remain visible. Private GPU and
complete HTTP/lifecycle qualification are separate from this CPU/source review.
