import collections,sys
best=collections.defaultdict(list)
for line in open(sys.argv[1]):
    f=line.split()
    if len(f)<9 or f[8]!='us': continue
    name,K,N=f[0],int(f[1]),int(f[2]); r=int(f[5][1:])
    kern=f[6]; us=float(f[7])
    variant='_'.join(kern.split('_')[5:])
    best[(name,K,N,r)].append((us,variant))
for key in sorted(best, key=lambda k:(k[0],k[3])):
    v=sorted(best[key])
    print(f'{key[0]:12s} K={key[1]:5d} N={key[2]:5d} r={key[3]}  ' + '  '.join(f'{b}:{a:.1f}' for a,b in v[:4]))
