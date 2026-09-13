"""Bounded weight-only typo consistency with frozen clean representation retention."""
import argparse,hashlib,json,time
from pathlib import Path
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from training.train_expanded import SparseCache,Menus,Development,read,validate_rows
from training.experiment_navigation import NavigationDev,navigation_pairs
from training.evaluate_expanded import load_artifact,Evaluator
from training.features import tokens
from training.model import encode_numpy
from training.export import export

class TypoDev:
    def __init__(self,rows):
        self.rows=rows;validate_rows(rows)
        self.cache=SparseCache([t for r in rows for t in [r['query'],*[c['label'] for c in r['candidates']]]])
        self.menus=Menus(rows,self.cache)
        self.short=[i for i,r in enumerate(rows) if len(tokens(r['cleanLabel']))<=2]
        self.long=[i for i in range(len(rows)) if i not in self.short]
    def measure(self,params,numpy=False):
        v=encode_numpy(params,self.cache.texts) if numpy else self.cache.all_vectors(params)
        scores=np.sum(v[self.menus.q,None,:]*v[self.menus.c],axis=-1)
        ranks=[sorted(range(len(r['candidates'])),key=lambda j:(-float(s[j]),j)) for r,s in zip(self.rows,scores)]
        def result(ids):
            hits=[self.rows[i]['relevance'].get(self.rows[i]['candidates'][ranks[i][0]]['id'],0)>=2 for i in ids]
            return {'queries':len(ids),'labels':len({self.rows[i]['cleanLabel'] for i in ids}),'top1':float(np.mean(hits)) if hits else 0.}
        out={'all':result(list(range(len(self.rows)))),'short':result(self.short),'long':result(self.long)}
        out['selectionScore']=.5*(out['all']['top1']+out['short']['top1']);return out

def passes(e,n,cfg):
    return e['devOverallNdcg5']>=cfg['minimumExpandedOverall'] and e['devSemanticNdcg5']>=cfg['minimumExpandedSemantic'] and e['devSemanticCoverage']>=cfg['minimumExpandedSemanticCoverage'] and e['devNoMatchSemanticRate']<=cfg['maximumNoMatchRate'] and n['semanticOnlyRaw']['ndcg5']>=cfg['minimumXfceRawSemantic']

def run(cfg,weight,seed,rows,navpairs,typo_pairs,expanded,navdev,typodev,initial):
    texts=[t for r in rows for t in [r['query'],*[c['label'] for c in r['candidates']]]]+[t for pair in navpairs+typo_pairs for t in pair]
    cache=SparseCache(texts);menus=Menus(rows,cache)
    navids=np.array([[cache.lookup[a],cache.lookup[b]] for a,b in navpairs],np.int32)
    typoids=np.array([[cache.lookup[a],cache.lookup[b]] for a,b in typo_pairs],np.int32)
    p={k:mx.array(v) for k,v in initial.items()};teacher=mx.array(cache.all_vectors(p));mx.eval(teacher)
    optimizer=optim.AdamW(learning_rate=cfg['learningRate'],weight_decay=cfg['weightDecay'])
    rng=np.random.default_rng(seed);prng=np.random.default_rng(seed+1000);mx.random.seed(seed)
    def loss(params,unique,qi,ci,pos,judged,ni,ti):
        v=cache.encode(params,unique);sim=mx.sum(v[qi,None,:]*v[ci],axis=-1)/cfg['temperature']
        ce=mx.mean(mx.logsumexp(mx.where(judged,sim,-1e9),axis=1)-mx.logsumexp(mx.where(pos,sim,-1e9),axis=1))
        nv=cache.encode(params,ni.reshape(-1)).reshape((-1,2,16));nc=mx.sum(nv[:,0,:]*nv[:,1,:],axis=-1)
        total=ce+cfg['navigationWeight']*mx.mean(mx.maximum(0,cfg['navigationTarget']-nc)**2)
        cleanids=mx.concatenate([unique,ni.reshape(-1),ti[:,1]])
        current=cache.encode(params,cleanids);tc=mx.sum(current*teacher[cleanids],axis=-1)
        total=total+cfg['teacherWeight']*mx.mean((1-tc)**2)
        if weight:
            tv=cache.encode(params,ti.reshape(-1)).reshape((-1,2,16));consistency=mx.sum(tv[:,0,:]*mx.stop_gradient(tv[:,1,:]),axis=-1)
            total=total+weight*mx.mean((1-consistency)**2)
        return total
    grad=mx.value_and_grad(loss);suffix='-'+cfg['tag'] if cfg.get('tag') else ''
    out=Path('runs')/f"typo-consistency{str(weight).replace('.','p')}-seed{seed}{suffix}"
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);history=[];best=None;bestscore=-1.;start=time.perf_counter()
    for epoch in range(cfg['epochs']+1):
        losses=[]
        if epoch:
            order=rng.permutation(len(rows))
            for offset in range(0,len(rows),cfg['batchSize']):
                ni=navids[prng.integers(len(navids),size=cfg['batchSize'])];ti=typoids[prng.integers(len(typoids),size=cfg['batchSize'])]
                value,g=grad(p,*menus.batch(order[offset:offset+cfg['batchSize']]),mx.array(ni),mx.array(ti));g,_=optim.clip_grad_norm(g,cfg['clipNorm']);p=optimizer.apply_gradients(g,p);mx.eval(value,p,optimizer.state);losses.append(float(value))
        e=expanded.measure(p);n=navdev.measure(p,e['cutoff']);t=typodev.measure(p);gate=passes(e,n,cfg)
        entry={'epoch':epoch,'loss':float(np.mean(losses)) if losses else None,'expanded':e,'xfce':n,'typoDev':t,'passesGates':gate};history.append(entry)
        if gate and t['selectionScore']>bestscore:
            best=entry;bestscore=t['selectionScore'];values={k:np.array(v) for k,v in p.items()}
            if not all(np.isfinite(v).all() for v in values.values()):raise FloatingPointError('nonfinite weights')
            np.savez(out/'weights.npz',**values)
        print(out.name,epoch,'short/all',round(t['short']['top1'],4),round(t['all']['top1'],4),'exp/cov/fp',round(e['devSemanticNdcg5'],4),round(e['devSemanticCoverage'],4),round(e['devNoMatchSemanticRate'],4),'nav',round(n['semanticOnlyRaw']['ndcg5'],4),gate,flush=True)
    meta={'architecture':'pooled','dimension':16,'family':'both','seed':seed,'typoWeight':weight,'configuration':cfg,'selected':best,'history':history,'trainingSeconds':time.perf_counter()-start,'status':'experimental candidate; no holdout read' if best else 'NOT SELECTED: no passing epoch','typoTrainPairs':len(typo_pairs),'anchorRows':len(rows),'navigationPairs':len(navpairs),'teacherScope':'Frozen initial embeddings of sampled anchor texts/navigation positive texts/canonical typo labels; no explicit noisy-variant teacher matching'}
    (out/'training.json').write_text(json.dumps(meta,indent=2)+'\n');return out,meta

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/typo-training.json');ap.add_argument('--tag');args=ap.parse_args();cfg=json.loads(Path(args.config).read_text())
    if args.tag:
        if not args.tag.replace('-','').isalnum():ap.error('invalid tag')
        cfg['tag']=args.tag
    paths=[*[Path(cfg['anchorData'])/f'{s}.jsonl' for s in ['train','dev','calibration']],Path(cfg['navigationTrain']),Path(cfg['navigationDev']),Path(cfg['typoTrain']),Path(cfg['typoDev']),Path(cfg['initialArtifact'])/'weights.bin',Path(cfg['initialArtifact'])/'manifest.json']
    cfg['sourceHashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    anchor,dev,cal,navtrain,navrows=map(read,paths[:5]);rows=[r for r in anchor if any(g>=2 for g in r['relevance'].values())];validate_rows(rows+dev+cal)
    tp=json.loads(paths[5].read_text());tr=read(paths[6]);initial,m=load_artifact(cfg['initialArtifact'])
    if m['featureVersion']!='gpu-search-features-v1' or any(v.shape!=(1024,16) for v in initial.values()):raise ValueError('requires currentv1 pooled16')
    expanded=Development(dev,cal);navdev=NavigationDev(navrows);typodev=TypoDev(tr);independent=Evaluator(cal,dev)
    report={'status':'Development-only weight consistency experiment; no typo holdout or diagnostic evaluated','configuration':cfg,'devShortCount':len(typodev.short),'devLongCount':len(typodev.long),'runs':[]}
    rp=Path('eval')/('typo-training'+('-'+args.tag if args.tag else '')+'.json')
    for weight,seed in cfg['runs']:
        out,meta=run(cfg,weight,seed,rows,navigation_pairs(navtrain),tp,expanded,navdev,typodev,initial);item={k:v for k,v in meta.items() if k!='history'};item['checkpoint']=str(out)
        if meta['selected']:
            artifact=Path('packages/model/experiments')/out.name
            if artifact.exists():raise FileExistsError(artifact)
            export(out,8,artifact);qp,qm=load_artifact(artifact)
            e=independent.evaluate(qp);n=navdev.measure(qp,e['cutoff'],numpy=True);t=typodev.measure(qp,numpy=True)
            gate=e['overall']['ndcg5']>=cfg['minimumExpandedOverall'] and e['semanticNdcg5']>=cfg['minimumExpandedSemantic'] and e['semanticCoverage']>=cfg['minimumExpandedSemanticCoverage'] and e['noMatchSemanticRate']<=cfg['maximumNoMatchRate'] and n['semanticOnlyRaw']['ndcg5']>=cfg['minimumXfceRawSemantic']
            item.update(artifact=str(artifact),payloadSha256=qm['payloadSha256'],payloadBytes=qm['payloadBytes'],int8Expanded=e,int8Xfce=n,int8TypoDev=t,int8PassesGates=gate)
        report['runs'].append(item)
        if any(hashlib.sha256(p.read_bytes()).hexdigest()!=cfg['sourceHashes'][str(p)] for p in paths):raise RuntimeError('input changed during frozen run')
        rp.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
