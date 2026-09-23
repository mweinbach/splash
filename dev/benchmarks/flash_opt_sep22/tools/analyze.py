import json,sys,collections
path=sys.argv[1]
rows=[json.loads(l) for l in open(path)]
# classify commands
cls=collections.defaultdict(list)
for r in rows:
    d=r.get('d',[])
    names=[x[0] for x in d]
    key=(r['n'], names[0] if names else r.get('first'))
    cls[key].append(r)
print('command classes (count, n dispatches, first pipeline, mean gpu_ms, mean sum dispatch ms, status)')
for key,rs in sorted(cls.items(), key=lambda kv:-len(kv[1])*kv[1][0]['gpu_ms']):
    sums=[sum(x[5] for x in r['d'] if x[5]>0)/1000 for r in rs if r.get('d')]
    print(len(rs), key[0], key[1], round(sum(r['gpu_ms'] for r in rs)/len(rs),3), round(sum(sums)/len(sums),3) if sums else None, rs[0].get('status'))
