import sys,re,collections
path=sys.argv[1]; start=int(sys.argv[2]); end=int(sys.argv[3])
lines=[]
for i,line in enumerate(open(path)):
    if i+1<start or i+1>end: continue
    m=re.match(r'^(\s*[+!:| ]*)(\d+)\s+(.*)$', line.rstrip())
    if not m: continue
    depth=len(m.group(1)); cnt=int(m.group(2)); name=m.group(3)
    name=re.sub(r'\(in [^)]*\)','',name); name=re.sub(r'\s+\+ \d+.*','',name)
    for a,b in [('splash::flash::',''),('std::__1::',''),('splash::metal::',''),('(anonymous namespace)::','')]: name=name.replace(a,b)
    name=re.sub(r'\(.*','',name)
    lines.append((depth,cnt,name))
# compute self samples (count minus children counts)
selfc=collections.Counter(); path_stack=[]
for idx,(d,c,n) in enumerate(lines):
    child=0; j=idx+1
    while j<len(lines) and lines[j][0]>d:
        if lines[j][0]==min(l[0] for l in lines[idx+1:j+1]) : pass
        j+=1
    # children = next-level entries directly below
    kids=[l for l in lines[idx+1:j] if l[0]==(lines[idx+1][0] if idx+1<j else -1)]
    selfc[n]+=c-sum(k[1] for k in kids)
# inclusive for key functions: sum top-most occurrences
def incl(name):
    tot=0; i=0
    while i<len(lines):
        d,c,n=lines[i]
        if n==name:
            tot+=c; j=i+1
            while j<len(lines) and lines[j][0]>d: j+=1
            i=j
        else: i+=1
    return tot
total=lines[0][1]
print('total',total)
for k in ['__psynch_cvwait','Worker::tick','FlashForward::forwardImpl','FlashMTPForward::forwardImpl','FlashForward::prefetchPLE','FlashPLESSD::prepare','FlashPLESSD::prefetch','Worker::emitTokens','Worker::publishStatus','Worker::safePoint','greedyRow','FlashForward::commitVerify','MetalBackend::submitCommandAsync','FlashMTPForward::truncate','Worker::copyHidden','dispatch_group_wait','Transport::wait','Worker::run']:
    print(f'{incl(k):6d} {k}')
print('--- top self')
for k,v in selfc.most_common(25): print(v,k[:120])
