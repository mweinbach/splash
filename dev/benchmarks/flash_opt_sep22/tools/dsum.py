import json,sys
for f in sys.argv[1:]:
    d=json.load(open(f'/Users/mweinbach/Projects/splash/build/opt-sep22/{f}.json'))['decode_suite']
    a=d['all_deltas']; c=a['mtp.verification_cycles']
    g=lambda k: round(a.get(k,0)/c,3)
    print(f"{f:8s} {d['aggregate_decode_tokens_per_second']:6.1f} tok/s tok/cyc {d['tokens_per_cycle']:.3f} cycle {g('metrics.decode_wall_ms')} verify wall {g('mtp.target_verify.forward_host_wall_ms')} gpu {g('mtp.target_verify.total_gpu_ms')} head wall {g('mtp.head_decode.forward_host_wall_ms')} gpu {g('mtp.head_decode.total_gpu_ms')} restore {g('mtp.prefix_restore.forward_host_wall_ms')} ple {g('ple_storage.host_read_ms')}")
