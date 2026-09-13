"""Bounded 32KiB supervised ablations; never modifies the baseline artifacts.

No-match targets use TRAIN's weak all-zero judgments only. Hard-negative margin
uses the hardest eligible judged negative already present in the TRAIN menu, not
unjudged labels mined from development or a global label inventory.
"""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import yaml

from training.train_expanded import SparseCache, Menus, Development, read, validate_rows
from training.evaluate import ndcg
from training.features import normalize


def mlx_objective(sim,pos,judged,arm):
    include_none=arm.get('no_match_weight',0)>0
    has_pos=mx.any(pos,axis=1)
    logits=sim/arm['temperature']
    contrast=mx.logsumexp(mx.where(judged,logits,-1e9),axis=1)-mx.logsumexp(mx.where(pos,logits,-1e9),axis=1)
    objective=mx.sum(mx.where(has_pos,contrast,0))/mx.maximum(mx.sum(has_pos),1)
    if arm.get('hard_margin_weight',0):
        best_negative=mx.max(mx.where(judged & ~pos,sim,-1e9),axis=1)
        positive_mean=mx.sum(mx.where(pos,sim,0),axis=1)/mx.maximum(mx.sum(pos,axis=1),1)
        hard=mx.maximum(0,arm['hard_margin']-positive_mean+best_negative)
        objective=objective+arm['hard_margin_weight']*mx.sum(mx.where(has_pos,hard,0))/mx.maximum(mx.sum(has_pos),1)
    if include_none:
        strongest=mx.max(mx.where(judged,sim,-1e9),axis=1)
        penalty=mx.maximum(0,strongest-arm['no_match_target'])**2
        objective=objective+arm['no_match_weight']*mx.sum(mx.where(~has_pos,penalty,0))/mx.maximum(mx.sum(~has_pos),1)
    return objective


class SlicedDevelopment(Development):
    def measure(self, params):
        result = super().measure(params)
        vectors = self.cache.all_vectors(params)
        scores = np.sum(vectors[self.dev.q,None,:]*vectors[self.dev.c],axis=-1)
        ranked=[];semantic_return=[]
        for i,row in enumerate(self.rows):
            pairs=[];semantic_return.append(False)
            for j,c in enumerate(row['candidates']):
                key=self.lex[i][j]
                if not key[0] and len(normalize(row['query']))>=3 and scores[i,j]>=result['cutoff']:
                    key=(1,0,float(scores[i,j]));semantic_return[-1]=True
                if key[0]:pairs.append((key,j))
            ranked.append([j for _,j in sorted(pairs,key=lambda pair:(tuple(-x for x in pair[0]),pair[1]))])
        slices={}
        for label in sorted({r.get('evaluation_slice','unspecified') for r in self.rows}):
            ids=[i for i,r in enumerate(self.rows) if r.get('evaluation_slice','unspecified')==label]
            sem=[i for i in ids if i in self.semantic]
            none=[i for i in ids if i in self.no_match]
            slices[label]={'count':len(ids),'semanticCount':len(sem),'semanticNdcg5':float(np.mean([ndcg(self.rows[i],ranked[i]) for i in sem])) if sem else None,'semanticCoverage':float(np.mean([bool(ranked[i]) for i in sem])) if sem else None,'noMatchCount':len(none),'noMatchSemanticRate':float(np.mean([semantic_return[i] for i in none])) if none else None}
        result['slices']=slices
        return result


def train(cfg, arm, seed, all_rows, development):
    include_none=arm.get('no_match_weight',0)>0
    rows=[r for r in all_rows if include_none or any(g>=2 for g in r['relevance'].values())]
    validate_rows(rows)
    cache=SparseCache([t for r in rows for t in [r['query'],*[c['label'] for c in r['candidates']]]])
    menus=Menus(rows,cache)
    grouped=defaultdict(list)
    for i,r in enumerate(rows):grouped[r['source_id']].append(i)
    groups=[np.array(grouped[k]) for k in sorted(grouped)]
    mx.random.seed(seed);rng=np.random.default_rng(seed)
    dimension=cfg['dimension']
    params={'word':mx.random.normal((1024,dimension))*.1,'char':mx.random.normal((1024,dimension))*.1}
    optimizer=optim.AdamW(learning_rate=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    def loss(p,unique,qi,ci,pos,judged):
        vectors=cache.encode(p,unique)
        sim=mx.sum(vectors[qi,None,:]*vectors[ci],axis=-1)
        return mlx_objective(sim,pos,judged,arm)
    value_grad=mx.value_and_grad(loss)
    suffix='-'+cfg['tag'] if cfg.get('tag') else ''
    out=Path('runs')/f"experiment-{arm['id']}-seed{seed}{suffix}"
    if out.exists():raise FileExistsError(f'Immutable experiment exists: {out}')
    out.mkdir(parents=True)
    best_key=(-1.,-1.,-1.);best_epoch=0;best_metrics=None;stale=0;history=[]
    start=time.perf_counter();optimizer_seconds=0.;seen_sources=defaultdict(int);seen_positive=0;seen_none=0
    for epoch in range(1,cfg['epochs']+1):
        n=cfg['epoch_examples']
        if arm.get('balanced'):
            # Uniform source mixture, then uniform row within that source; training only.
            chosen=rng.integers(len(groups),size=n)
            order=np.array([rng.choice(groups[g]) for g in chosen])
        else:
            order=np.concatenate([rng.permutation(len(rows)) for _ in range((n+len(rows)-1)//len(rows))])[:n]
        losses=[];step_start=time.perf_counter()
        for offset in range(0,n,cfg['batch_size']):
            selected=order[offset:offset+cfg['batch_size']]
            batch=menus.batch(selected)
            value,gradients=value_grad(params,*batch)
            gradients,_=optim.clip_grad_norm(gradients,cfg['clip_norm'])
            params=optimizer.apply_gradients(gradients,params)
            mx.eval(value,params,optimizer.state)
            losses.append(float(value))
        optimizer_seconds+=time.perf_counter()-step_start
        for i in order:
            seen_sources[rows[i]['source_id']]+=1
            if menus.pos[i].any():seen_positive+=1
            else:seen_none+=1
        metrics=development.measure(params)
        feasible=metrics['devNoMatchSemanticRate']<=cfg['max_dev_no_match_rate']
        key=(int(feasible),metrics['devSemanticNdcg5'],metrics['devRelevantCoverage'])
        snapshot={k:np.array(v) for k,v in params.items()}
        if not all(np.isfinite(v).all() for v in snapshot.values()):
            raise FloatingPointError('Refusing to save nonfinite model weights')
        if epoch in (1,10):
            ep=out/f'epoch-{epoch:03d}';ep.mkdir();np.savez(ep/'weights.npz',**snapshot)
            (ep/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        if key>best_key:
            best_key=key;best_epoch=epoch;best_metrics=metrics;stale=0
            np.savez(out/'weights.npz',**snapshot)
        else:stale+=1
        history.append({'epoch':epoch,'optimizerSteps':epoch*((n+cfg['batch_size']-1)//cfg['batch_size']),'loss':float(np.mean(losses)),'passesDevNoMatchGate':feasible,**metrics,'elapsedSeconds':time.perf_counter()-start})
        print(out.name,epoch,round(metrics['devSemanticNdcg5'],4),round(metrics['devNoMatchSemanticRate'],4),flush=True)
        if epoch>=cfg['minimum_epochs'] and stale>=cfg['patience']:break
    metadata={'architecture':'pooled','dimension':dimension,'family':'both','tensorParameters':2048*dimension,'int8WeightBytes':2048*dimension,'seed':seed,'arm':arm,'configuration':cfg,'bestEpoch':best_epoch,'epochs':len(history),'selectedPassesDevNoMatchGate':bool(best_key[0]),'bestMetrics':best_metrics,'trainingSeconds':time.perf_counter()-start,'optimizerSeconds':optimizer_seconds,'timingScope':'Training loop incl development/checkpointing; excludes feature/cache preparation','history':history,'sampledSourceCounts':dict(seen_sources),'sampledPositiveCount':seen_positive,'sampledNoMatchCount':seen_none,'uniqueInputRows':len(rows),'positiveRows':sum(menus.pos.any(axis=1)).item(),'noMatchRows':sum(~menus.pos.any(axis=1)).item(),'uniqueTexts':len(cache.texts),'truncatedTexts':cache.truncated,'status':('experimental candidate; passes dev no-match gate; original transfer dev not used' if best_key[0] else 'NOT SELECTED: no checkpoint passed dev no-match gate; retained for failed-run analysis'),'selection':'prefer <=5% dev no-match; then highest post-calibration semantic nDCG and coverage; earliest exact ties','hardNegativePolicy':'only judged negatives already in train menu; no global unjudged-negative mining','noMatchLoss':'mean squared hinge above fixed cosine target on train all-zero menus' if include_none else 'disabled','framework':'MLX 0.31.1','device':str(mx.default_device())}
    (out/'training.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return {'checkpoint':str(out),'arm':arm['id'],'seed':seed,'bestEpoch':best_epoch,'epochs':len(history),'passesDevNoMatchGate':bool(best_key[0]),'trainingSeconds':metadata['trainingSeconds'],'optimizerSeconds':optimizer_seconds,'weightsSha256':hashlib.sha256((out/'weights.npz').read_bytes()).hexdigest(),**best_metrics}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/experiments-training.yaml');parser.add_argument('--arm');parser.add_argument('--seed',type=int);parser.add_argument('--tag');args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    if cfg['dimension']!=16:raise ValueError('This experiment freezes the 32KiB architecture')
    if args.tag:
        if not args.tag.replace('-','').replace('_','').isalnum():parser.error('invalid tag')
        cfg['tag']=args.tag
    root=Path(cfg['data_dir']);paths=[root/f'{s}.jsonl' for s in ['train','dev','calibration']]
    cfg['sourceHashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows,dev,cal=map(read,paths);validate_rows(dev+cal)
    evaluation=SlicedDevelopment(dev,cal)
    report_path=Path('eval')/('training-experiments'+('-'+args.tag if args.tag else '')+'.json')
    report=json.loads(report_path.read_text()) if report_path.exists() else {'status':'Weak-label development ablations, no sealed test claim','configuration':cfg,'runs':[]}
    arms=[a for a in cfg['arms'] if not args.arm or a['id']==args.arm]
    if not arms:parser.error('unknown arm')
    for arm in arms:
        for seed in [args.seed] if args.seed is not None else cfg['seeds']:
            result=train(cfg,arm,seed,rows,evaluation);report['runs'].append(result)
            if any(hashlib.sha256(p.read_bytes()).hexdigest()!=cfg['sourceHashes'][str(p)] for p in paths):raise RuntimeError('Data changed during experiment')
            report_path.write_text(json.dumps(report,indent=2)+'\n')
    groups=defaultdict(list)
    for run in report['runs']:groups[run['arm']].append(run)
    report['summary']={arm:{'seeds':len(rs),'passingSeeds':sum(r['passesDevNoMatchGate'] for r in rs),'meanSemanticNdcg5':float(np.mean([r['devSemanticNdcg5'] for r in rs])),'minSemanticNdcg5':min(r['devSemanticNdcg5'] for r in rs),'maxDevNoMatchRate':max(r['devNoMatchSemanticRate'] for r in rs),'meanHeldFamilyNdcg5':float(np.mean([r['slices']['unseen-intent-family']['semanticNdcg5'] for r in rs])),'meanUiNamespaceNdcg5':float(np.mean([r['slices']['ui-settings-unseen-namespace']['semanticNdcg5'] for r in rs]))} for arm,rs in groups.items()}
    report_path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
