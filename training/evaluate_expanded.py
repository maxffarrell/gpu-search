"""Development-only comparison. Calibration chooses cutoffs; dev chooses checkpoints.

Never opens a sealed test. Original pilot dev is used only by the final reporting CLI.
"""
import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from training.features import features, normalize, tokens
from training.evaluate import lexical, ndcg, osa
from training.export import unpack


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def calibrate(rows, scores, maximum_rate=.05):
    maxima = [max(s, default=-np.inf) for row, s in zip(rows, scores) if not any(row['relevance'].values())]
    if not maxima:
        return 1.01
    # Strictly above the (allowed + 1)th largest value. Ties abstain conservatively.
    ordered = sorted(maxima, reverse=True)
    index = min(len(ordered)-1, math.floor(maximum_rate * len(ordered)))
    value = ordered[index]
    return float(np.nextafter(np.float32(value), np.float32(np.inf))) if np.isfinite(value) else 1.01


def summarize(rows, rankings, tiers=None):
    positive = [i for i,r in enumerate(rows) if any(g > 0 for g in r['relevance'].values())]
    relevant = [i for i,r in enumerate(rows) if any(g >= 2 for g in r['relevance'].values())]
    none = [i for i,r in enumerate(rows) if not any(r['relevance'].values())]
    mean = lambda values: float(np.mean(values)) if values else None
    result = {'count':len(rows),'positiveCount':len(positive),'relevantCount':len(relevant),'noMatchCount':len(none),
              'ndcg5':mean([ndcg(rows[i],rankings[i]) for i in positive]),
              'coverage':mean([bool(rankings[i]) for i in relevant]),
              'noMatchAnyRate':mean([bool(rankings[i]) for i in none]),
              'noMatchSemanticRate':mean([any(tiers is None or not tiers[i][j][0] for j in rankings[i]) for i in none])}
    for k in [1,3,5]:
        result[f'recall{k}']=mean([sum(rows[i]['relevance'].get(rows[i]['candidates'][j]['id'],0)>=2 for j in rankings[i][:k])/sum(g>=2 for g in rows[i]['relevance'].values()) for i in relevant])
    result['mrr']=mean([next((1/(n+1) for n,j in enumerate(rankings[i]) if rows[i]['relevance'].get(rows[i]['candidates'][j]['id'],0)>=2),0) for i in relevant])
    return result


def rankings(rows, scores, cutoff, tiers, raw=False):
    result=[]
    for row, values, lexical_tiers in zip(rows,scores,tiers):
        ranked=[]
        for i,score in enumerate(values):
            tier=(0,0,0) if raw else lexical_tiers[i]
            if tier[0]: ranked.append((tier,i))
            elif len(normalize(row['query']))>=3 and np.isfinite(score) and score>=cutoff: ranked.append(((1,0,float(score)),i))
        result.append([i for _,i in sorted(ranked,key=lambda pair:(tuple(-x for x in pair[0]),pair[1]))])
    return result


class Evaluator:
    """Cache bounded hashed features and hard tiers once for fast checkpoint selection."""
    def __init__(self, calibration_rows, dev_rows):
        self.calibration=calibration_rows
        self.rows=dev_rows
        self.all_rows=calibration_rows+dev_rows
        self.tiers=[[lexical(row['query'],c) for c in row['candidates']] for row in dev_rows]
        self.semantic=[i for i,row in enumerate(dev_rows) if any(g>=2 for g in row['relevance'].values()) and not any(t[0] and row['relevance'].get(c['id'],0)>=2 for t,c in zip(self.tiers[i],row['candidates']))]
        texts={r['query'] for r in self.all_rows}
        for row in self.all_rows:
            for c in row['candidates']: texts.update([c['label'],*c.get('aliases',[]),c.get('context','')])
        self.feature_cache={text:features(text) for text in texts}

    def scores(self,params,family='both'):
        vectors={}
        for text,f in self.feature_cache.items():
            x=np.zeros(params['word'].shape[1],np.float32)
            for kind,key in [('word','wordIds'),('char','charIds')]:
                if f[key] and family in ('both',kind): x+=np.float32(.5)*params[kind][f[key]].mean(axis=0)
            h=np.tanh(x@params['W1'].T+params['b1']) if 'W1' in params else x
            z=h@params['W2'].T+params['b2'] if 'W2' in params else h
            norm=np.linalg.norm(z)
            vectors[text]=z/norm if norm>=1e-8 else np.zeros_like(z)
        composed={}
        output=[]
        for row in self.all_rows:
            q=vectors[row['query']]
            scores=[]
            for c in row['candidates']:
                key=(c['label'],tuple(c.get('aliases',[])),c.get('context',''))
                if key not in composed:
                    vs=[vectors[t] for t in [c['label'],*c.get('aliases',[])] if np.linalg.norm(vectors[t])>=1e-8]
                    v=np.mean(vs,axis=0) if vs else np.zeros_like(q)
                    if c.get('context'):v=v+.25*vectors[c['context']]
                    norm=np.linalg.norm(v);composed[key]=v/norm if norm>=1e-8 else np.zeros_like(v)
                v=composed[key]
                scores.append(float(q@v) if np.linalg.norm(q)>=1e-8 and np.linalg.norm(v)>=1e-8 else -np.inf)
            output.append(scores)
        return output

    def evaluate(self,params,family='both',detail=False):
        all_scores=self.scores(params,family)
        cutoff=calibrate(self.calibration,all_scores[:len(self.calibration)])
        scores=all_scores[len(self.calibration):]
        ranked=rankings(self.rows,scores,cutoff,self.tiers)
        overall=summarize(self.rows,ranked,self.tiers)
        semantic=summarize([self.rows[i] for i in self.semantic],[ranked[i] for i in self.semantic],[self.tiers[i] for i in self.semantic])
        result={'cutoff':cutoff,'semanticNdcg5':semantic['ndcg5'] or 0.,'semanticCoverage':semantic['coverage'] or 0.,'overallNdcg5':overall['ndcg5'] or 0.,'noMatchSemanticRate':overall['noMatchSemanticRate'],'overall':overall,'semanticOnly':semantic}
        if detail:result.update(scores=scores,rankings=ranked)
        return result


def load_artifact(directory):
    path=Path(directory);m=json.loads((path/'manifest.json').read_text());payload=(path/'weights.bin').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=m['payloadSha256']:raise ValueError('artifact hash mismatch')
    params={t['name']:(unpack(payload[t['byteOffset']:t['byteOffset']+t['byteLength']],t['elementCount'],t['bits']).reshape(t['shape'])*np.float32(t['scale'])).astype(np.float32) for t in m['tensors']}
    return params,m


def bootstrap(rows,left,right,indices,seed=20260913):
    delta=np.array([ndcg(rows[i],left[i])-ndcg(rows[i],right[i]) for i in indices])
    if not len(delta):return {'count':0,'delta':None,'pairedBootstrap95':None}
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(5000)])
    return {'count':len(delta),'delta':float(delta.mean()),'pairedBootstrap95':np.quantile(means,[.025,.975]).tolist()}


def train_aliases(rows):
    aliases={}
    for row in rows:
        for c in row['candidates']:
            if row['relevance'].get(c['id'],0)>=2: aliases.setdefault(c['label'],set()).add(row['query'])
    return {label:sorted(values) for label,values in aliases.items()}


@lru_cache(maxsize=None)
def lexical_field(text):
    return normalize(text),tokens(text)


def alias_lexical(query,candidate,aliases):
    """Same matcher as lexical(), with impossible OSA lengths pruned before allocation."""
    q,qt=lexical_field(query)
    if not q:return (0,0,0)
    if q==normalize(candidate['label']):return (3,0,1)
    best=(0,0,0)
    for index,text in enumerate([candidate['label'],*candidate.get('aliases',[]),*aliases.get(candidate['label'],[])]):
        f,ft=lexical_field(text);score=min(1,len(q.replace(' ',''))/max(1,len(f.replace(' ',''))))
        if index and q==f:rank=(2,5,1)
        elif len(q)>=2 and f.startswith(q):rank=(2,4,score)
        else:
            cursor=0
            for token in ft:
                if cursor<len(qt) and token.startswith(qt[cursor]):cursor+=1
            if qt and cursor==len(qt) and len(q.replace(' ',''))>=3:rank=(2,3,score)
            elif 2<=len(q)<=6 and len(ft)>=2 and q==''.join(t[0] for t in ft):rank=(2,2,1)
            else:
                longest=max(len(q),len(f));bound=1 if longest<=7 else 2
                if min(len(q),len(f))<4 or abs(len(q)-len(f))>bound:continue
                distance=osa(q,f)
                rank=(2,1,1-distance/longest) if distance<=bound and distance/longest<=.25 else (0,0,0)
        best=max(best,rank)
    return best


class Tfidf:
    """Train-only vocabulary/IDF, querying candidates and optional train paraphrases."""
    def __init__(self,train):
        self.aliases=train_aliases(train)
        texts=sorted({t for r in train for c in r['candidates'] for t in [c['label'],*c.get('aliases',[]),c.get('context','')]} | {r['query'] for r in train})
        counts={}
        for text in texts:
            for g in set(self.grams(text)):counts[g]=counts.get(g,0)+1
        self.idf={g:math.log((1+len(texts))/(1+n))+1 for g,n in counts.items()}
    @staticmethod
    def grams(text):
        t='^'+normalize(text)+'$'
        return ['c:'+t[i:i+n] for n in [2,3,4] for i in range(len(t)-n+1)]+['w:'+word for word in tokens(text)]
    @lru_cache(maxsize=None)
    def vector(self,text):
        result={}
        for g in self.grams(text):
            if g in self.idf:result[g]=result.get(g,0)+self.idf[g]
        norm=math.sqrt(sum(x*x for x in result.values()))
        return {g:v/norm for g,v in result.items()} if norm else {}
    def score(self,rows,paraphrases=False):
        output=[]
        for row in rows:
            q=self.vector(row['query']);scores=[]
            for c in row['candidates']:
                fields=[c['label'],*c.get('aliases',[])]
                if c.get('context'):fields.append(c['context'])
                if paraphrases:fields+=self.aliases.get(c['label'],[])
                best=0.
                for text in fields:
                    field=self.vector(text)
                    smaller,larger=(q,field) if len(q)<len(field) else (field,q)
                    best=max(best,sum(v*larger.get(g,0) for g,v in smaller.items()))
                scores.append(best)
            output.append(scores)
        return output


def report(train,calibration,dev,old_directory,new_directory):
    evaluator=Evaluator(calibration,dev)
    systems={};ranked={};all_scores={};cutoffs={}
    empty=[[-np.inf]*len(r['candidates']) for r in dev]
    ranked['lexical']=rankings(dev,empty,1.01,evaluator.tiers)
    aliases=train_aliases(train)
    alias_tiers=[[alias_lexical(r['query'],c,aliases) for c in r['candidates']] for r in dev]
    ranked['lexical+all-train-aliases']=rankings(dev,empty,1.01,alias_tiers)
    tf=Tfidf(train)
    for name,use_alias in [('train-tfidf',False),('nearest-train-paraphrase',True)]:
        cal=tf.score(calibration,use_alias);scores=tf.score(dev,use_alias);cutoff=calibrate(calibration,cal)
        ranked[name]=rankings(dev,scores,cutoff,evaluator.tiers);cutoffs[name]=cutoff
    for name,directory in [('old-int8',old_directory),('new-int8',new_directory)]:
        p,m=load_artifact(directory);result=evaluator.evaluate(p,m['featureFamily'],detail=True)
        scores=result['scores'];ranked[name]=result['rankings'];cutoffs[name]=result['cutoff']
        ranked[name+'-raw']=rankings(dev,scores,-1,evaluator.tiers,raw=True)
        ranked[name+'-raw-cutoff']=rankings(dev,scores,result['cutoff'],evaluator.tiers,raw=True)
        ranked[name+'-combined-before-cutoff']=rankings(dev,scores,-1,evaluator.tiers)
        all_scores[name]={'modelId':m['modelId'],'sha256':m['payloadSha256']}
    for name,indices in ranked.items():
        tiers=alias_tiers if name=='lexical+all-train-aliases' else None if '-raw' in name else evaluator.tiers
        systems[name]={'overall':summarize(dev,indices,tiers),'semanticOnly':summarize([dev[i] for i in evaluator.semantic],[indices[i] for i in evaluator.semantic],None if tiers is None else [tiers[i] for i in evaluator.semantic])}
        systems[name]['slices']={}
        for slice_name in sorted({r.get('evaluation_slice','pilot-transfer') for r in dev}):
            selected=[i for i,r in enumerate(dev) if r.get('evaluation_slice','pilot-transfer')==slice_name]
            systems[name]['slices'][slice_name]=summarize([dev[i] for i in selected],[indices[i] for i in selected],None if tiers is None else [tiers[i] for i in selected])
    baselines=['lexical','lexical+all-train-aliases','train-tfidf','nearest-train-paraphrase']
    strongest=max(baselines,key=lambda name:systems[name]['semanticOnly']['ndcg5'] or 0.)
    comparisons={name:bootstrap(dev,ranked['new-int8'],ranked[name],evaluator.semantic) for name in ['old-int8',strongest]}
    gates=json.loads(Path('configs/gates.json').read_text())
    comparison=comparisons[strongest]
    provisional=bool(comparison['delta'] is not None and comparison['delta']>=gates['semanticNdcg5Delta'] and comparison['pairedBootstrap95'][0]>gates['pairedBootstrap95LowerMustExceed'] and (systems['new-int8']['overall']['noMatchSemanticRate'] or 0)<=gates['maximumNoMatchSemanticRate'] and systems['new-int8']['overall']['ndcg5']>=systems[strongest]['overall']['ndcg5']-gates['maximumOverallRegression'])
    return {'scope':'Weakly labeled development; seen-intent and unseen transfer reported separately; no sealed test execution','trainCount':len(train),'calibrationCount':len(calibration),'devCount':len(dev),'semanticCount':len(evaluator.semantic),'cutoffSource':'expanded/calibration only','models':all_scores,'cutoffs':cutoffs,'systems':systems,'strongestBaseline':strongest,'comparisons':comparisons,'frozenGates':gates,'provisionalNumericGatesPassed':provisional,'releaseGatePassed':False,'releaseGateReason':'Reviewed independent final evaluation required; dev selected checkpoint','rowIds':[r['id'] for r in dev],'rankings':{name:[indices[:5] for indices in runs] for name,runs in ranked.items()}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--old',required=True);parser.add_argument('--new',required=True);parser.add_argument('--data',default='data/expanded');args=parser.parse_args()
    root=Path(args.data);train=read_rows(root/'train.jsonl');cal=read_rows(root/'calibration.jsonl');dev=read_rows(root/'dev.jsonl')
    result=report(train,cal,dev,args.old,args.new)
    Path('eval/expanded-evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    # Post-selection only: this distribution never enters checkpoint selection or training.
    transfer=report(train,cal,read_rows('data/dev.jsonl'),args.old,args.new)
    transfer['scope']='Original pilot dev: post-selection out-of-domain transfer regression only'
    Path('eval/expanded-transfer.json').write_text(json.dumps(transfer,indent=2)+'\n')
    quantized,metadata=load_artifact(args.new)
    checkpoint=Path('runs')/metadata['modelId']/'weights.npz'
    if checkpoint.exists():
        evaluator=Evaluator(cal,dev)
        floating=evaluator.evaluate(dict(np.load(checkpoint)),detail=True)
        integer=evaluator.evaluate(quantized,detail=True)
        fixed=rankings(dev,integer['scores'],floating['cutoff'],evaluator.tiers)
        semantic=evaluator.semantic
        fixed_metric=summarize([dev[i] for i in semantic],[fixed[i] for i in semantic])['ndcg5']
        loss=floating['semanticNdcg5']-integer['semanticNdcg5']
        fixed_loss=floating['semanticNdcg5']-fixed_metric
        quality={'float':{k:v for k,v in floating.items() if k not in ['scores','rankings']},'int8':{k:v for k,v in integer.items() if k not in ['scores','rankings']},'semanticNdcgLoss':loss,'overallNdcgLoss':floating['overallNdcg5']-integer['overallNdcg5'],'fixedFloatCutoffSemanticNdcgLoss':fixed_loss,'maximumPermittedLoss':.01,'passed':max(loss,fixed_loss)<=.01,'cutoffsIndependentlyCalibrated':True}
        Path('eval/expanded-quantization.json').write_text(json.dumps(quality,indent=2)+'\n')
        audits=[]
        for data_kind in ['more-data','old-data-control']:
            for seed in [17,29,43]:
                base=Path(f'runs/expanded-{data_kind}-pooled-seed{seed}')
                for variant in ['epoch-001','epoch-010','']:
                    directory=base/variant
                    if (directory/'weights.npz').exists():
                        audits.append({'path':str(directory),'seed':seed,'data':data_kind,'metrics':evaluator.evaluate(dict(np.load(directory/'weights.npz')))})
        Path('eval/expanded-checkpoint-audit.json').write_text(json.dumps({'method':'Independent NumPy pooled encoder, expanded calibration cutoff, expanded dev only','originalPilotDevUsed':False,'runs':audits},indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rankings'},indent=2))


if __name__=='__main__':main()
