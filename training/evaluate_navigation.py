"""Provider-held-out documented-positive retrieval; never invent negative labels."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from training.evaluate_expanded import Evaluator, Tfidf, read_rows, rankings, calibrate, ndcg
from training.evaluate_release_candidate import asset_metadata, model_scores
from training.features import normalize


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(rows,ranks,indices):
    if not indices:return {'queries':0}
    result={'queries':len(indices),'ndcg5':float(np.mean([ndcg(rows[i],ranks[i]) for i in indices])),
            'coverage':float(np.mean([bool(ranks[i]) for i in indices])),
            'noMatchRate':None,'judgments':'Only documented positives; other destinations are unjudged.'}
    for k in [1,3,5]:
        result[f'success{k}']=float(np.mean([any(rows[i]['relevance'].get(rows[i]['candidates'][j]['id'],0)>=2 for j in ranks[i][:k]) for i in indices]))
    return result


def paired(rows,left,right,indices):
    differences=np.array([ndcg(rows[i],left[i])-ndcg(rows[i],right[i]) for i in indices])
    if not len(differences):return {'count':0}
    groups=defaultdict(list)
    for offset,i in enumerate(indices):groups[tuple(sorted(rows[i]['relevance']))].append(offset)
    clusters=list(groups.values());rng=np.random.default_rng(20260913)
    independent=[];clustered=[]
    for _ in range(3000):
        independent.append(rng.choice(differences,len(differences),replace=True).mean())
        sample=np.concatenate([clusters[j] for j in rng.integers(len(clusters),size=len(clusters))])
        clustered.append(differences[sample].mean())
    return {'count':len(indices),'documentedDestinationGroups':len(clusters),'delta':float(differences.mean()),
            'queryBootstrap95':np.quantile(independent,[.025,.975]).tolist(),
            'destinationGroupBootstrap95':np.quantile(clustered,[.025,.975]).tolist(),
            'scope':'One provider and partial source labels; intervals do not establish population-wide or best-in-class quality.'}


def teacher_scores(rows):
    # Optional offline quality reference; none of these libraries are browser dependencies.
    import torch
    from transformers import AutoModel, AutoTokenizer
    name='sentence-transformers/all-MiniLM-L6-v2';revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
    tokenizer=AutoTokenizer.from_pretrained(name,revision=revision,local_files_only=True)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model=AutoModel.from_pretrained(name,revision=revision,local_files_only=True,use_safetensors=True).to(device).eval()
    texts=sorted({text for row in rows for text in [row['query'],*[c['label'] for c in row['candidates']]]})
    vectors=[]
    with torch.inference_mode():
        for start in range(0,len(texts),128):
            batch=tokenizer(texts[start:start+128],padding=True,truncation=True,max_length=256,return_tensors='pt').to(device)
            values=model(**batch).last_hidden_state;mask=batch['attention_mask'].unsqueeze(-1)
            pooled=(values*mask).sum(1)/mask.sum(1).clamp(min=1)
            vectors.extend(torch.nn.functional.normalize(pooled,dim=1).cpu().numpy())
    lookup=dict(zip(texts,vectors))
    return [[float(lookup[row['query']]@lookup[c['label']]) for c in row['candidates']] for row in rows],{'id':name,'revision':revision,'weightBytes':90868376,'dimension':384,'shipped':False,'device':device}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);parser.add_argument('--partition',choices=['dev','holdout'],default='dev');parser.add_argument('--selection-record');parser.add_argument('--teacher',action='store_true');parser.add_argument('--output',default='eval/navigation-evaluation.json');args=parser.parse_args()
    metadata=asset_metadata(args.candidate)
    if metadata['payloadBytes']>32768:raise ValueError('Candidate exceeds current raw model budget')
    if args.partition=='holdout':
        if not args.selection_record:raise ValueError('Holdout requires a frozen selection record')
        selection=json.loads(Path(args.selection_record).read_text())
        if selection.get('selectionFrozen') is not True or any(selection.get(k)!=metadata[k] for k in ['payloadSha256','manifestSha256']):raise ValueError('Selection is not frozen for this exact candidate')
        receipt={'accessedAt':datetime.now(timezone.utc).isoformat(),'selection':args.selection_record,'selectionSha256':digest(args.selection_record),'candidate':metadata,'partition':'data/navigation-splits/holdout.jsonl','scope':'One frozen evaluation event; provider holdout, not fully reviewed negatives.'}
        with Path('eval/navigation-holdout-access.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    path=f'data/navigation-splits/{args.partition}.jsonl';rows=read_rows(path)
    calibration=read_rows('data/expanded/calibration.jsonl')
    train=read_rows('data/expanded/train.jsonl')+read_rows('data/navigation-splits/train.jsonl')
    train_queries={normalize(row['query']) for row in train}
    evaluator=Evaluator([],rows)
    slices={'all':list(range(len(rows))),'semanticOnly':evaluator.semantic,
            'seenTrainingQuery':[i for i,r in enumerate(rows) if normalize(r['query']) in train_queries],
            'newTrainingQuery':[i for i,r in enumerate(rows) if normalize(r['query']) not in train_queries]}
    scores={};cutoffs={};artifacts={}
    for key,artifact in [('live','packages/model/candidate'),('prior-research','packages/model/candidate-v2'),('candidate',args.candidate)]:
        scores[key]=model_scores(artifact,rows)
        cutoffs[key]=calibrate(calibration,model_scores(artifact,calibration))
        artifacts[key]=asset_metadata(artifact)
    tfidf=Tfidf(train)
    for key,paraphrases in [('tfidf',False),('train-paraphrase-tfidf',True)]:
        scores[key]=tfidf.score(rows,paraphrases)
        cutoffs[key]=calibrate(calibration,tfidf.score(calibration,paraphrases))
    teacher=None
    if args.teacher:
        combined,teacher=teacher_scores(calibration+rows)
        scores['teacher']=combined[len(calibration):];cutoffs['teacher']=calibrate(calibration,combined[:len(calibration)])
    results={};ranks={}
    lexical=rankings(rows,[[-np.inf]*len(r['candidates']) for r in rows],1.01,evaluator.tiers)
    results['lexical']={name:metrics(rows,lexical,ids) for name,ids in slices.items()}
    for key,values in scores.items():
        for mode in ['raw','calibrated']:
            label=f'{key}-{mode}'
            ranks[label]=rankings(rows,values,cutoffs[key] if mode=='calibrated' else -np.inf,evaluator.tiers,raw=mode=='raw')
            results[label]={name:metrics(rows,ranks[label],ids) for name,ids in slices.items()}
    comparisons={}
    for key in scores:
        if key=='candidate':continue
        for mode in ['raw','calibrated']:
            comparisons[f'candidate-minus-{key}-{mode}']=paired(rows,ranks[f'candidate-{mode}'],ranks[f'{key}-{mode}'],slices['semanticOnly'])
    report={'scope':'Known-positive retrieval on source-authored software search keywords; absence is not a negative judgment.',
            'partition':path,'partitionSha256':digest(path),'artifacts':artifacts,'teacher':teacher,'cutoffs':cutoffs,'cutoffFit':'Existing expanded calibration only; no navigation threshold fitting.',
            'slices':{name:len(ids) for name,ids in slices.items()},'metrics':results,'comparisons':comparisons,
            'sourceHashes':{p:digest(p) for p in ['data/expanded/train.jsonl','data/expanded/calibration.jsonl','data/navigation-splits/train.jsonl','training/evaluate_navigation.py']},
            'negativeJudgmentsAvailable':False,'productionReadinessEstablished':False,'bestInClassEstablished':False,
            'perQuery':[{'id':r['id'],'query':r['query'],'knownPositives':r['relevance'],'top5':{key:rank[i][:5] for key,rank in ranks.items()}} for i,r in enumerate(rows)]}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'candidate':metadata['modelId'],'slices':report['slices'],'raw':results['candidate-raw']['semanticOnly'],'calibrated':results['candidate-calibrated']['semanticOnly'],'delta':comparisons['candidate-minus-live-raw'],'output':args.output},indent=2))


if __name__=='__main__':main()
