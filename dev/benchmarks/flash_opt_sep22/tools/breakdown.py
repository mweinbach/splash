import json,sys,collections
path=sys.argv[1]; n=int(sys.argv[2]); top=int(sys.argv[3]) if len(sys.argv)>3 else 60
rows=[json.loads(l) for l in open(path)]
rs=[r for r in rows if r['n']==n and r.get('d')]
agg=collections.defaultdict(lambda:[0,0.0,set(),0])
for r in rs:
    for name,gx,gy,gz,t,us,b in r['d']:
        a=agg[name]; a[0]+=1; a[1]+=us; a[2].add((gx,gy,gz,t)); a[3]=max(a[3],b)
N=len(rs); tot=sum(a[1] for a in agg.values())/N
print(f'{N} commands, mean sum {tot/1000:.3f} ms, mean gpu {sum(r["gpu_ms"] for r in rs)/N:.3f}')
print(f'{"pipeline":60s} {"calls":>6s} {"ms":>8s} {"us/call":>8s} {"%":>5s}  grids')
for name,a in sorted(agg.items(), key=lambda kv:-kv[1][1])[:top]:
    print(f'{name[:60]:60s} {a[0]/N:6.1f} {a[1]/N/1000:8.3f} {a[1]/a[0]:8.1f} {100*a[1]/N/tot:5.1f}  {sorted(a[2])[:3]} maxbind={a[3]/1e6:.1f}MB')
