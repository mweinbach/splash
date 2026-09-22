Private pure-I8 Teacher lease-only experiment, September22

Parent: build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5. This worker retains the entire
parent Forward body, constructor, F32 presence/lookup policy, all508 actual F32
mappings, all509 BF16 mappings, every shader and all backing/planner/governor
charges. Its numerical derivative remains b87448342df3b3a9ae1642379b8aab513bb208ff234e552bcf70efc26a82c09d.

SPLASH_FLASH_TEACHER_SINGLETON_LEASE_PRUNE_SEP22=0 defaults to original saved owner
selection:1281 owners /145924161536 bytes. Flag1 changes only copied startup lease
selection: fixedR4 F32118/3247964160 plus BF16425/4692377600 plus Full51296/121174228992
plus already-present W8derived168/1890975744 =807/131005546496.390F32/11143741440 and
84BF16/3774873600 are omitted only from persistent registration. They remain
mapped, owned, charged and available to existing direct command bindings. No
physical pinning or hardware schedule guarantee is made. Public lease destruction
is not a detach API; no phase swapping was introduced.

All batch capabilities, math, allocations and flags remain parent-identical. The
proposed resource profile is qualified only for B1 canonical benchmarking; it
does not claim batch throughput qualification. Numerical and task quality cannot
be inherited from prior phaseQ4 resource experiments.

CPU-only preparation/build/witness:
  .venv/bin/python -B dev/benchmarks/teacher_singleton_lease_sep22/census_plan.py
  .venv/bin/python -B dev/benchmarks/teacher_singleton_lease_sep22/prepare.py --output FRESH_PRIVATE_BUILD
  make -f FRESH_PRIVATE_BUILD/machinery/worker.mk BUILD=FRESH_PRIVATE_BUILD -j8 all
  .venv/bin/python -B dev/benchmarks/teacher_singleton_lease_sep22/witness.py --build FRESH_PRIVATE_BUILD

Root exclusively owns GPU execution/model payload access. The exact canonical
command is root-model-command.txt: all Parent flags unchanged, explicit new flag1,
MTP3, canonical2048 prompt/256 output, B1,1warm/3trials, context16384 and frozen22
semantic plan/unchanged graders. Use a fresh output path. Startup source-bound
profile, owner census and fresh governor evidence are mandatory before interpreting
performance. CPU fixtures contain actual saved Parent outputs with synthetic807
metadata; they prove only validators and original accounting, no new runtime or
performance evidence.
