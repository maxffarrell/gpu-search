"""Isolated bucket/dimension tradeoffs at <=32768 int8 bytes.

Retains deployed bigram25 mixing. No signed hashing, objective changes, source
balance or new training data are mixed into this structural experiment.
"""
import argparse,hashlib,json,struct,time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from training.features import tokens,features as unigram_features
from training.features_bigram import features as deployed_features
from training.train_expanded import Menus,Development,read,validate_rows
from training.evaluate_expanded import calibrate,rankings,summarize
from training.export import quantize


def raw_hash(blob):
    h=2166136261
    for byte in blob:h=((h^byte)*16777619)&0xffffffff
    return h


@lru_cache(maxsize=None)
def blobs(text):
    ts=[t[:64] for t in tokens(text)[:32]]
    words=[('w:'+t).encode() for t in ts];bigrams=[];chars=[]
    for a,b in zip(ts,ts[1:]):
        aa,bb=a.encode(),b.encode();bigrams.append(b'b:'+len(aa).to_bytes(4,'little')+aa+len(bb).to_bytes(4,'little')+bb)
    for t in ts:
        elems=[b'\x01']+[b'\x00'+ord(c).to_bytes(4,'little') for c in t]+[b'\x02']
        for n in [2,3,4]:
            for i in range(len(elems)-n+1):
                if len(chars)<512:chars.append(b'c:'+b''.join(elems[i:i+n]))
    return tuple(words),tuple(bigrams),tuple(chars)


@lru_cache(maxsize=None)
def raw_features(text):
    return tuple(tuple(map(raw_hash,group)) for group in blobs(text))


def weighted_features(text,arm):
    words,bigrams,chars=raw_features(text)
    word_ids=[h&(arm['wordBuckets']-1) for h in words+bigrams]
    char_ids=[h&(arm['charBuckets']-1) for h in chars]
    mixing=.25 if bigrams else 0.
    ww=[.5*(1-mixing)/len(words)]*len(words) if words else []
    if bigrams:ww += [.5*mixing/len(bigrams)]*len(bigrams)
    cw=[.5/len(chars)]*len(chars) if chars else []
    return word_ids,ww,char_ids,cw


class BucketCache:
    def __init__(self,texts,arm):
        self.texts=list(dict.fromkeys(texts));self.lookup={s:i for i,s in enumerate(self.texts)};self.arm=arm
        fs=[weighted_features(s,arm) for s in self.texts];self.numpy=[]
        for ik,wk in [(0,1),(2,3)]:
            width=max(1,max(len(f[ik]) for f in fs));ids=np.zeros((len(fs),width),np.int32);weights=np.zeros((len(fs),width),np.float32)
            for i,f in enumerate(fs):ids[i,:len(f[ik])]=f[ik];weights[i,:len(f[wk])]=f[wk]
            self.numpy.extend([ids,weights])
        self.arrays=list(map(mx.array,self.numpy));mx.eval(*self.arrays)
    def encode(self,p,indices):
        wi,ww,ci,cw=self.arrays
        x=mx.sum(p['word'][wi[indices]]*ww[indices,:,None],axis=1)+mx.sum(p['char'][ci[indices]]*cw[indices,:,None],axis=1)
        return x/mx.maximum(mx.sqrt(mx.sum(x*x,axis=-1,keepdims=True)),1e-8)
    def all_vectors(self,p):
        output=[]
        for start in range(0,len(self.texts),512):
            v=self.encode(p,mx.arange(start,min(start+512,len(self.texts))));mx.eval(v);output.append(np.array(v))
        return np.concatenate(output)
    def numpy_vectors(self,p):
        # Independent NumPy reduction, using per-text feature occurrence lists.
        output=[]
        for text in self.texts:
            wi,ww,ci,cw=weighted_features(text,self.arm)
            x=np.zeros(self.arm['dimension'],np.float32)
            for name,ids,weights in [('word',wi,ww),('char',ci,cw)]:
                if ids:x+=np.sum(p[name][ids]*np.array(weights,np.float32)[:,None],axis=0,dtype=np.float32)
            norm=np.linalg.norm(x);output.append(x/max(norm,1e-8))
        return np.array(output)


def scores(rows,cache,vectors):
    out=[]
    for row in rows:
        q=vectors[cache.lookup[row['query']]];line=[]
        for c in row['candidates']:
            v=vectors[cache.lookup[c['label']]];norm=np.linalg.norm(v)
            v=v/max(norm,1e-8) # single-label candidate composition final normalization
            line.append(float(q@v) if norm>=1e-8 and np.linalg.norm(q)>=1e-8 else -np.inf)
        out.append(line)
    return out


def independent(cache,params,cal,dev,tiers,semantic,fixed_cutoff=None):
    v=cache.numpy_vectors(params);cs=scores(cal,cache,v);ds=scores(dev,cache,v)
    cutoff=calibrate(cal,cs) if fixed_cutoff is None else fixed_cutoff
    ranks=rankings(dev,ds,cutoff,tiers)
    sem=summarize([dev[i] for i in semantic],[ranks[i] for i in semantic],[tiers[i] for i in semantic])
    overall=summarize(dev,ranks,tiers)
    slices={}
    for label in sorted({r.get('evaluation_slice','unspecified') for r in dev}):
        ids=[i for i in semantic if dev[i].get('evaluation_slice','unspecified')==label]
        slices[label]=summarize([dev[i] for i in ids],[ranks[i] for i in ids],[tiers[i] for i in ids])
    return {'cutoff':cutoff,'semanticNdcg5':sem['ndcg5'],'overall':overall,'semanticOnly':sem,'slices':slices,'passesDevNoMatchGate':overall['noMatchSemanticRate']<=.05},ds


def export_research(folder,params,arm,contract):
    out=Path('packages/model/experiments')/folder.name;out.mkdir(parents=True,exist_ok=False)
    packed=bytearray();tensors=[];quantized={}
    for name in ['word','char']:
        w=params[name];q,scale=quantize(w,8);blob=q.tobytes()
        tensors.append({'name':name,'shape':list(w.shape),'elementCount':w.size,'bits':8,'scale':float(scale),'scaleF32LE':struct.pack('<f',scale).hex(),'byteOffset':len(packed),'byteLength':len(blob)})
        packed.extend(blob);quantized[name]=(q.astype(np.float32)*scale).astype(np.float32)
    assert len(packed)<=32768 and len(packed)==arm['int8Bytes']
    manifest={'formatVersion':'gpu-search-collision-research-v1','featureVersion':arm['featureVersion'],'architecture':'pooled','dimension':arm['dimension'],'featureFamily':'both','modelId':folder.name,'tensors':tensors,'payloadBytes':len(packed),'payloadSha256':hashlib.sha256(packed).hexdigest(),'byteOrder':'little-endian','validatedSemanticCutoff':None,'status':'research only; baseline runtime unchanged','contract':arm}
    (out/'weights.bin').write_bytes(packed);(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'feature-spec.json').write_text(json.dumps(contract,indent=2)+'\n')
    return out,quantized


def run(arm,seed,cfg,contract,train,dev,cal,development):
    suffix='-'+cfg['tag'] if cfg.get('tag') else ''
    folder=Path('runs')/f"collision-{arm['id']}-seed{seed}{suffix}"
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True)
    rows=[r for r in train if any(g>=2 for g in r['relevance'].values())]
    cache=BucketCache([t for r in rows for t in [r['query'],*[c['label'] for c in r['candidates']]]],arm)
    menus=Menus(rows,cache);development.cache=BucketCache(development.cache.texts,arm)
    mx.random.seed(seed);rng=np.random.default_rng(seed);d=arm['dimension']
    p={'word':mx.random.normal((arm['wordBuckets'],d))*.1,'char':mx.random.normal((arm['charBuckets'],d))*.1}
    optimizer=optim.AdamW(learning_rate=cfg['learningRate'],weight_decay=cfg['weightDecay'])
    def loss(params,unique,qi,ci,pos,judged):
        v=cache.encode(params,unique);sim=mx.sum(v[qi,None,:]*v[ci],axis=-1)/cfg['temperature']
        return mx.mean(mx.logsumexp(mx.where(judged,sim,-1e9),axis=1)-mx.logsumexp(mx.where(pos,sim,-1e9),axis=1))
    grad=mx.value_and_grad(loss);best=None;bestkey=(-1.,-1.);stale=0;history=[];start=time.perf_counter()
    for epoch in range(1,cfg['maxEpochs']+1):
        order=rng.permutation(len(rows));losses=[]
        for offset in range(0,len(rows),cfg['batchSize']):
            value,g=grad(p,*menus.batch(order[offset:offset+cfg['batchSize']]));g,_=optim.clip_grad_norm(g,cfg['clipNorm']);p=optimizer.apply_gradients(g,p);mx.eval(value,p,optimizer.state);losses.append(float(value))
        metrics=development.measure(p);key=(metrics['devSemanticNdcg5'],metrics['devRelevantCoverage']);eligible=metrics['devNoMatchSemanticRate']<=cfg['maximumDevFalsePositiveRate']
        entry={'epoch':epoch,'loss':float(np.mean(losses)),**metrics};history.append(entry)
        if eligible and key>bestkey:
            best=entry;bestkey=key;stale=0
            weights={k:np.array(v) for k,v in p.items()}
            if not all(np.isfinite(v).all() for v in weights.values()):raise FloatingPointError('nonfinite checkpoint')
            np.savez(folder/'weights.npz',**weights)
        else:stale+=1
        if epoch in [1,10]:np.savez(folder/f'epoch-{epoch:03d}.npz',**{k:np.array(v) for k,v in p.items()})
        print(folder.name,epoch,round(metrics['devSemanticNdcg5'],4),round(metrics['devNoMatchSemanticRate'],4),flush=True)
        if epoch>=cfg['minimumEpochs'] and stale>=cfg['patience']:break
    meta={'arm':arm,'seed':seed,'configuration':cfg,'history':history,'selected':best,'epochs':len(history),'trainingSeconds':time.perf_counter()-start,'timingScope':'Materialized MLX loop incl dev evaluation; excludes cache construction; concurrent research may contend for GPU','status':'experimental dev-selected checkpoint' if best else 'NOT SELECTED: no epoch passed dev no-match gate','trainingRows':len(rows),'noPositiveRowsExcluded':len(train)-len(rows)}
    if best:
        weights=dict(np.load(folder/'weights.npz'));reference,float_scores=independent(development.cache,weights,cal,dev,development.lex,development.semantic)
        artifact,quantized=export_research(folder,weights,arm,contract)
        quant,fixed_scores=independent(development.cache,quantized,cal,dev,development.lex,development.semantic,reference['cutoff'])
        calibrated,_=independent(development.cache,quantized,cal,dev,development.lex,development.semantic)
        raw_gpu=development.cache.all_vectors({k:mx.array(v) for k,v in weights.items()});raw_numpy=development.cache.numpy_vectors(weights)
        meta.update(artifact=str(artifact),independentFloat=reference,int8FixedCutoff=quant,int8Recalibrated=calibrated,quantizationNdcgLoss=reference['semanticNdcg5']-quant['semanticNdcg5'],maxVectorParityError=float(np.max(np.abs(raw_gpu-raw_numpy))))
        fixture_texts=['','Profile','my information','delete account','C++ C#','myProfile','repeated repeated']
        fc=BucketCache(fixture_texts,arm);fv=fc.numpy_vectors(quantized)
        (artifact/'fixtures.json').write_text(json.dumps([{'text':t,'features':weighted_features(t,arm),'vector':fv[i].tolist()} for i,t in enumerate(fixture_texts)],indent=2)+'\n')
    (folder/'training.json').write_text(json.dumps(meta,indent=2)+'\n')
    return {k:v for k,v in meta.items() if k!='history'}


def audit(train,arms):
    texts={r['query'] for r in train}|{c['label'] for r in train for c in r['candidates']}
    word=set();char=set()
    for t in texts:
        w,b,c=blobs(t);word.update(w+b);char.update(c)
    output={}
    for arm in arms:
        counts={}
        for kind,preimages,buckets in [('wordIncludingBigrams',word,arm['wordBuckets']),('char',char,arm['charBuckets'])]:
            groups=defaultdict(int)
            for b in preimages:groups[raw_hash(b)&(buckets-1)]+=1
            counts[kind]={'distinctPreimages':len(preimages),'occupiedBuckets':len(groups),'bucketsWithCollisions':sum(n>1 for n in groups.values()),'distinctFeaturesBeyondFirst':len(preimages)-len(groups),'maximumBucketLoad':max(groups.values())}
        output[arm['id']]=counts
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/collision-training.json');parser.add_argument('--contracts',default='configs/collision-contracts.json');parser.add_argument('--tag');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());contract=json.loads(Path(args.contracts).read_text())
    if args.tag:
        if not args.tag.replace('-','').isalnum():parser.error('invalid tag')
        cfg['tag']=args.tag
    paths=[Path(cfg['dataDir'])/f'{s}.jsonl' for s in ['train','dev','calibration']];cfg['sourceHashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    cfg['contractSha256']=hashlib.sha256(Path(args.contracts).read_bytes()).hexdigest()
    train,dev,cal=map(read,paths);validate_rows(train+dev+cal)
    # Prove control feature IDs reproduce the existing deployed contract.
    for text in [r['text'] for r in json.loads(Path('fixtures/features.json').read_text())]:
        w,b,c=raw_features(text);f=deployed_features(text)
        assert [h&1023 for h in w]==f['wordIds'] and [h&1023 for h in b]==f['bigramIds'] and [h&1023 for h in c]==f['charIds']
    report={'status':'Isolated bucket/dimension research; no source test access','configuration':cfg,'contracts':contract,'collisionAudit':audit(train,contract['arms']),'runs':[]}
    out=Path('eval')/('collision-tradeoffs'+('-'+args.tag if args.tag else '')+'.json');out.write_text(json.dumps(report,indent=2)+'\n')
    development=Development(dev,cal)
    for arm in contract['arms']:
        if (arm['wordBuckets']+arm['charBuckets'])*arm['dimension']!=arm['int8Bytes'] or arm['int8Bytes']>32768:raise ValueError('payload budget failure')
        for seed in cfg['seeds']:
            report['runs'].append(run(arm,seed,cfg,contract,train,dev,cal,development))
            if any(hashlib.sha256(p.read_bytes()).hexdigest()!=cfg['sourceHashes'][str(p)] for p in paths):raise RuntimeError('source data changed')
            out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
