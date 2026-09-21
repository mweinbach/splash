# Optional SSD PLE mode: launcher and idle maintenance

SSD streaming is an explicit launch option. The accepted v9 profile remains
unchanged at 38 static flags, and a normal launch keeps the PLE table in memory.

```
./splash serve --model local/Qwen3.8-Flash-Next-oQ4e-mtp \
  --local-package install/local-models/Flash-Next-oQ4e-mtp-v1 \
  --port 8011 --ple-ssd-streaming --ple-ssd-cache-mb 64
```

`--ple-ssd-streaming` sets `SPLASH_FLASH_PLE_SSD_STREAMING=1` before model startup.
`--ple-ssd-cache-mb` sets `SPLASH_FLASH_PLE_SSD_CACHE_MB`; it accepts a canonical
decimal size from 0 to 1024 MiB. The native store default is 64 MiB. A zero cache
still uses bounded I/O scratch. Cache options require streaming enabled, and both
options require `--local-model` or `--local-package`. Explicit CLI values override
the corresponding environment value; omitted values preserve the environment.
No source model, imported package, accepted profile or oMLX setting is changed.

Idle maintenance keeps the other weights warm using a separate exact reflected
readonly pointer layout. It never binds the SSD PLE table, the CPU row cache or
the mutable GPU row staging. Both paths still read one four-byte word per actual
immutable native owner and check each output, checksum, owner count and guards.

| Storage mode | Original native owners | Original native bytes | Complete owners | Complete native bytes |
| --- | ---: | ---: | ---: | ---: |
| Original in-memory PLE | 21 | 106,320,429,056 | 1,134 | 144,326,852,608 |
| SSD PLE, remaining original weights in memory | 28 | 74,317,889,536 | 1,141 | 112,324,313,088 |

The SSD union excludes 32,002,539,520 padded PLE bytes. Both unions include the
same 1,113 verified derived owners and 38,006,423,552 derived bytes. Geometry
validation still requires the exact source digest, layout digest, 48 layers,
512 experts and hidden width 2,560. Missing optional derived stores disable
maintenance with the existing status explanation; mismatched model or original
native geometry is rejected. The original `flash_idle_immutable_touch_v1` shader
body and scheduler behavior are unchanged. The SSD entry point reflects exactly
1,141 non-PLE pointer slots; it uses no dummy or null owner.

CPU verification completed for this bounded integration:

- 66 launcher/profile tests, including seven SSD option and environment tests.
- 8,258 SSD/raw geometry and output checks; 7,951 original policy checks;
  37,842 original scheduler checks.
- Three source inventory checks for exact reflected layouts, explicit Worker
  mode and source ownership, and backend pointer-count validation.
- Metal 4.1 shader compile and C++ maintenance header compile with warnings
  treated as errors.

These checks contain no GPU execution. Full loader, model parity, idle warming,
service quality and performance qualification belong to the coordinated root
run; this note alone does not claim those passed.
