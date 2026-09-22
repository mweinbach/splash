# Independent CPU review

Reviewer: `/root/decode_kernel/pref_oracle_snapshot_review`, with a bounded script/closure audit by `snapshot_closure_cautions`.

Release artifacts:

- Teacher exporter: `build/trunkpref-exact-teacher-sep22-v3`, oracle `bbd6e9c3fce10fe2e910d68ac56d34c3f52704f0bf2e6bd37573422325fa2c0e`.
- Phase comparator: `build/trunkpref-exact-phase-sep22-v2`, oracle `16b2bcf29d1e4f5453fd20df19696d3b93373a9f7880a7adcf2e76950b72d4fe`.
- Exact shared metallib: `bb09bf88bb53a8b6e9bfc5254068c16942bb0913784ff1c672d810f13c6eb6f0`.
- Phase-v3 worker CPU seal: `cb44ee7757168b07779a085f65b627897ce6b5e86562bed123658b7984c15a66`.
- Final comparator command: `6d3d070a578f25b3a54523cd13e0d14e599ca56f589b7452f92cacd91f8d77f6`.

The intended teacher/export and phase/compare commands have no fatal source or closure blocker. These are CPU review and serializer results; only Root-owned executions can produce the actual byte-preservation evidence.

The audit confirms every one of the 134 physical trunk-state planes, scalar state metadata, and complete returned hidden/logits/greedy records. Every borrowed output is consumed before the next forward call. Direct 1 MiB streaming comparisons include framing and physical payload bytes and reject mismatches, truncation, extent differences, trailing bytes and nonfinite numeric words. The inventory is 14 state/output checkpoints, 28 unique frames, and 12 repeated-body frame comparisons; the bounded spill fits 4 GiB. The phase graph census is exactly 960 I8 Prefill calls and 693,360 encoded rows, with zero Decode/Verify/Q4-Prefill work.

Each oracle uses its corresponding worker's exact effective closure: 54 linked worker objects, with only the Worker object containing `main` excluded, leaving 53 nonworker objects. Forward and `teacher_bulk` are retained, normalized duplicate identities are rejected, headers are unmodified, source include roots are copied, and timing ABI remains 200. The teacher exporter has 281 source entries and the comparator 283.

The final comparator was diffed against the running teacher-v3 exporter. Only diagnostic resource/report changes were added; serialization, frame labels, input schedule, metadata and common-policy construction remain unchanged. All 38 shared numeric Prefill flags match in the generated commands. Consequently the running teacher export remains compatible and needs no repeat for these diagnostic changes.

The final comparator commits its diagnostic reservation, checks workspace plus one state against its plan, and records a final governor snapshot. It does not create the Worker's residency lease. Both trained-head/prime proof and Worker-residency proof remain explicitly false; no performance or service-lifecycle claim is made.

The final launcher validates its command against the ready seal, validates its own runner hash, rejects oracle/metallib drift against the compiled manifest, checks manifest/argv/environment role pairing, and requires successful completed export before comparison. The running exporter retains the original Root-inspected command and artifact pins. Most numeric body flags are recorded before backend creation and validated in their underlying constructors; the prepared launcher supplies the reviewed frozen profile and removes inherited Splash flags.

Reviewers made no edits, GPU calls, model/input payload reads or hashes. Artifact/source hashes and metadata inspection are distinct from input/model payload access.
