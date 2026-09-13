"""Current-v1 warm-start with source-authored navigation positive alignment.

Only paths explicitly named in configuration are read. Navigation missing grades
are unjudged: they never enter a negative loss. No aliases are used at prediction.
"""
import argparse,hashlib,json,time
from pathlib import Path
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from training.train_expanded import SparseCache,Menus,Development,read,validate_rows
from training.features import normalize
from training.evaluate import lexical,ndcg
from training.evaluate_expanded import load_artifact,Evaluator
from training.export import export


def known_metrics(rows,ranks):
    positive=[i for i,r in enumerate(rows) if any(g>=2 for g in r['relevance'].values())]
    mean=lambda xs:float(np.mean(xs)) if xs else None
    out={'queryCount':len(rows),'knownPositivePairs':sum(sum(g>=2 for g in r['relevance'].values()) for r in rows),'multiPositiveQueries':sum(sum(g>=2 for g in r['relevance'].values())>1 for r in rows),'ndcg5':mean([ndcg(rows[i],ranks[i]) for i in positive]),'coverage':mean([bool(ranks[i]) for i in positive]),'judgmentWarning':'Only upstream-listed destinations have known relevance. All other candidates are unjudged, not established negatives; metrics are known-positive diagnostics.'}
    for k in [1,5]:
        out[f'recall{k}']=mean([sum(rows[i]['relevance'].get(rows[i]['candidates'][j]['id'],0)>=2 for j in ranks[i][:k])/sum(g>=2 for g in rows[i]['relevance'].values()) for i in positive])
    return out


class NavigationDev:
    def __init__(self,rows):
        validate_rows(rows);self.rows=rows
        self.cache=SparseCache([t for r in rows for t in [r['query'],*[c['label'] for c in r['candidates']]]])
        self.menus=Menus(rows,self.cache)
        self.semantic=[i for i,r in enumerate(rows) if not any(lexical(r['query'],c)[0] and r['relevance'].get(c['id'],0)>=2 for c in r['candidates'])]
    def measure(self,params,cutoff,numpy=False):
        if numpy:
            from training.model import encode_numpy
            vectors=encode_numpy(params,self.cache.texts)
        else:vectors=self.cache.all_vectors(params)
        scores=np.sum(vectors[self.menus.q,None,:]*vectors[self.menus.c],axis=-1)
        raw=[];calibrated=[]
        for row,line in zip(self.rows,scores):
            ranked=sorted(range(len(row['candidates'])),key=lambda j:(-float(line[j]),j))
            raw.append(ranked);calibrated.append([j for j in ranked if len(normalize(row['query']))>=3 and line[j]>=cutoff])
        idx=self.semantic
        return {'raw':known_metrics(self.rows,raw),'calibrated':known_metrics(self.rows,calibrated),'semanticOnlyRaw':known_metrics([self.rows[i] for i in idx],[raw[i] for i in idx]),'semanticOnlyCalibrated':known_metrics([self.rows[i] for i in idx],[calibrated[i] for i in idx]),'cutoffSource':'expanded calibration only','cutoff':cutoff}


def navigation_pairs(rows):
    from training.navigation_augmentation import documented_positive_pairs
    return documented_positive_pairs(rows)


def eligible(metrics,cfg):
    return metrics['devOverallNdcg5']>=cfg.get('minimumExpandedOverallNdcg5',0.) and metrics['devSemanticNdcg5']>=cfg['minimumExpandedSemanticNdcg5'] and metrics['devRelevantCoverage']>=cfg['minimumExpandedRelevantCoverage'] and metrics['devNoMatchSemanticRate']<=cfg['maximumExpandedNoMatchRate']


def run(cfg,weight,seed,anchor,navtrain,expanded,navdev,initial,lexpairs=None):
    positive=[r for r in anchor if any(g>=2 for g in r['relevance'].values())]
    original_count=len(positive);shortened=[]
    if cfg.get('shortQueryAugmentation'):
        from training.navigation_augmentation import augment_positive_rows
        shortened=augment_positive_rows(positive,cfg['politePrefixes'])
        if not shortened:raise ValueError('No train-only shortened examples available')
        positive=positive+shortened
    pairs=navigation_pairs(navtrain)
    lexpairs=lexpairs or [];lexweight=cfg.get('lexiconWeight',0.)
    texts=[t for r in positive for t in [r['query'],*[c['label'] for c in r['candidates']]]]+[t for pair in pairs+lexpairs for t in pair]
    cache=SparseCache(texts);menus=Menus(positive,cache);pairids=np.array([[cache.lookup[q],cache.lookup[label]] for q,label in pairs],np.int32)
    lexpairids=np.array([[cache.lookup[q],cache.lookup[label]] for q,label in lexpairs],np.int32) if lexpairs else np.zeros((1,2),np.int32)
    mx.random.seed(seed);rng=np.random.default_rng(seed);navrng=np.random.default_rng(seed+1000);lexrng=np.random.default_rng(seed+2000)
    params={k:mx.array(v) for k,v in initial.items()}
    optimizer=optim.AdamW(learning_rate=cfg['learningRate'],weight_decay=cfg['weightDecay'])
    def loss(p,unique,qi,ci,pos,judged,alignment_ids,lexicon_ids):
        v=cache.encode(p,unique);sim=mx.sum(v[qi,None,:]*v[ci],axis=-1)/cfg['temperature']
        ce=mx.mean(mx.logsumexp(mx.where(judged,sim,-1e9),axis=1)-mx.logsumexp(mx.where(pos,sim,-1e9),axis=1))
        if weight==0 and lexweight==0:return ce
        # Every pair is explicitly positive; there is deliberately no negative denominator.
        objective=ce
        for ids,coefficient in [(alignment_ids,weight),(lexicon_ids,lexweight)]:
            if coefficient:
                av=cache.encode(p,ids.reshape(-1)).reshape((-1,2,16))
                cosine=mx.sum(av[:,0,:]*av[:,1,:],axis=-1)
                objective=objective+coefficient*mx.mean(mx.maximum(0,cfg['alignmentTarget']-cosine)**2)
        return objective
    grad=mx.value_and_grad(loss)
    slug=str(weight).replace('.','p');suffix='-'+cfg['tag'] if cfg.get('tag') else ''
    out=Path('runs')/f'navigation-align{slug}-seed{seed}{suffix}'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    history=[];best=None;bestkey=-1.;start=time.perf_counter()
    for epoch in range(cfg['epochs']+1):
        losses=[]
        if epoch:
            if shortened:
                # Exactly50:50 (up to one odd example) original and derived pools.
                nshort=original_count//2
                originals=rng.permutation(original_count)[:original_count-nshort]
                derived=rng.integers(original_count,len(positive),size=nshort)
                order=np.concatenate([originals,derived]);rng.shuffle(order)
            else:order=rng.permutation(len(positive))
            for offset in range(0,len(order),cfg['batchSize']):
                ids=order[offset:offset+cfg['batchSize']]
                selected=navrng.integers(len(pairs),size=cfg['navigationBatchSize'])
                lexselected=lexrng.integers(len(lexpairids),size=cfg['navigationBatchSize'])
                value,g=grad(params,*menus.batch(ids),mx.array(pairids[selected]),mx.array(lexpairids[lexselected]))
                g,_=optim.clip_grad_norm(g,cfg['clipNorm']);params=optimizer.apply_gradients(g,params);mx.eval(value,params,optimizer.state);losses.append(float(value))
        metrics=expanded.measure(params);nav=navdev.measure(params,metrics['cutoff']);gate=eligible(metrics,cfg)
        entry={'epoch':epoch,'loss':float(np.mean(losses)) if losses else None,'expanded':metrics,'navigationDev':nav,'passesExpandedGates':gate};history.append(entry)
        quality=nav['semanticOnlyRaw']['ndcg5']
        if gate and quality>bestkey:
            bestkey=quality;best=entry
            weights={k:np.array(v) for k,v in params.items()}
            if not all(np.isfinite(v).all() for v in weights.values()):raise FloatingPointError('nonfinite weights')
            np.savez(out/'weights.npz',**weights)
        print(out.name,epoch,round(quality,4),round(metrics['devSemanticNdcg5'],4),round(metrics['devRelevantCoverage'],4),round(metrics['devNoMatchSemanticRate'],4),'eligible',gate,flush=True)
    meta={'architecture':'pooled','dimension':16,'family':'both','featureVersion':'gpu-search-features-v1','seed':seed,'alignmentWeight':weight,'lexiconWeight':lexweight,'lexiconPositivePairs':len(lexpairs),'configuration':cfg,'selected':best,'history':history,'trainingSeconds':time.perf_counter()-start,'anchorPositiveRows':original_count,'shortenedPositiveRows':len(shortened),'anchorSampling':'50:50 original/derived pools, equal total update count; derived eligible subset is oversampled' if shortened else 'original positives only','navigationPositivePairs':len(pairs),'navigationRows':len(navtrain),'pairPolicy':'all explicit grade>=2 query-label pairs; missing grades excluded from alignment and never made negatives','status':'experimental candidate only; GNOME holdout not accessed' if best else 'NOT SELECTED: no checkpoint retained expanded gates','int8WeightBytes':32768}
    (out/'training.json').write_text(json.dumps(meta,indent=2)+'\n')
    return out,meta


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/navigation-training.json');p.add_argument('--tag');args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    if args.tag:
        if not args.tag.replace('-','').isalnum():p.error('invalid tag')
        cfg['tag']=args.tag
    paths=[Path(cfg['navigationTrain']),Path(cfg['navigationDev']),*[Path(cfg['anchorData'])/f'{s}.jsonl' for s in ['train','dev','calibration']],Path(cfg['initialArtifact'])/'manifest.json',Path(cfg['initialArtifact'])/'weights.bin']
    if cfg.get('lexiconPairs'):paths.extend([Path(cfg['lexiconPairs']),Path(cfg['lexiconManifest'])])
    cfg['sourceHashes']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    navtrain,navrows,anchor,dev,cal=[read(path) for path in paths[:5]]
    validate_rows(anchor+dev+cal)
    initial,manifest=load_artifact(cfg['initialArtifact'])
    if manifest['featureVersion']!='gpu-search-features-v1' or sum(p.size for p in initial.values())!=32768 or set(initial)!=set(['word','char']) or any(w.shape!=(1024,16) or not np.isfinite(w).all() for w in initial.values()):raise ValueError('initial artifact must be current32KiB v1 pooled16')
    lexpairs=[(r['left'],r['right']) for r in json.loads(Path(cfg['lexiconPairs']).read_text())] if cfg.get('lexiconPairs') else []
    if cfg.get('lexiconWeight',0) and not lexpairs:raise ValueError('Lexicon weight requires positive pairs')
    nav=NavigationDev(navrows);expanded=Development(dev,cal);independent=Evaluator(cal,dev)
    overlap=sorted({normalize(r['query']) for r in navtrain}&{normalize(r['query']) for r in navrows})
    report={'scope':'KDE documented keyword positives for alignment, XFCE dev only; no GNOME holdout or consumed source test access','configuration':cfg,'navigationTrainRows':len(navtrain),'navigationDevRows':len(navrows),'navigationTrainPositivePairs':len(navigation_pairs(navtrain)),'navigationDevPositivePairs':len(navigation_pairs(navrows)),'normalizedQueryOverlap':overlap,'queryOverlapCount':len(overlap),'partialJudgments':'Navigation unlisted candidates remain unjudged. Known-positive retrieval metrics do not establish false-positive rates or exhaustive relevance.','runs':[]}
    reportpath=Path('eval')/('navigation-training'+('-'+args.tag if args.tag else '')+'.json')
    for weight in cfg['weights']:
        for seed in cfg['seeds']:
            folder,meta=run(cfg,weight,seed,anchor,navtrain,expanded,nav,initial,lexpairs)
            item={k:v for k,v in meta.items() if k!='history'};item['checkpoint']=str(folder)
            if meta['selected']:
                artifact=Path('packages/model/experiments')/folder.name
                if artifact.exists():raise FileExistsError(artifact)
                export(folder,8,artifact)
                qp,qm=load_artifact(artifact)
                e=independent.evaluate(qp)
                qn=nav.measure(qp,e['cutoff'],numpy=True)
                item.update(artifact=str(artifact),int8Expanded=e,int8NavigationDev=qn,int8PassesExpandedGates=e['overall']['ndcg5']>=cfg.get('minimumExpandedOverallNdcg5',0.) and e['semanticNdcg5']>=cfg['minimumExpandedSemanticNdcg5'] and e['overall']['coverage']>=cfg['minimumExpandedRelevantCoverage'] and e['noMatchSemanticRate']<=cfg['maximumExpandedNoMatchRate'])
            report['runs'].append(item)
            if any(hashlib.sha256(path.read_bytes()).hexdigest()!=cfg['sourceHashes'][str(path)] for path in paths):raise RuntimeError('frozen source changed')
            reportpath.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
