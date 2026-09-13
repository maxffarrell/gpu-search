"""Evaluate frozen typo weights without lexical ranking or query corrections."""
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from training.evaluate_release_candidate import asset_metadata,model_scores
from training.evaluate_expanded import read_rows

BASELINE='packages/model/experiments/navigation-align0p5-seed29'
def evaluate(artifact,rows):
    scores=model_scores(artifact,rows)
    ranks=[sorted(range(len(s)),key=lambda j:(-s[j],j)) for s in scores]
    hits=[int(r['relevance'].get(r['candidates'][rank[0]]['id'],0)>=2) for r,rank in zip(rows,ranks)]
    return {'slices': {name: {'count':len(ids),'top1':float(np.mean([hits[i] for i in ids])) if ids else None} for name,ids in [('shortLabels',[i for i,r in enumerate(rows) if len(r['cleanLabel'].split())<=2]),('longLabels',[i for i,r in enumerate(rows) if len(r['cleanLabel'].split())>2])]}, 'top1':float(np.mean(hits)),'top5':float(np.mean([any(r['relevance'].get(r['candidates'][j]['id'],0)>=2 for j in rank[:5]) for r,rank in zip(rows,ranks)])),'perQuery':[{'id':r['id'],'cleanLabel':r['cleanLabel'],'query':r['query'],'hit':hit,'top5':[r['candidates'][j]['id'] for j in rank[:5]]} for r,rank,hit in zip(rows,ranks,hits)]}

def claim(selection,artifact,receipt):
    record=json.loads(Path(selection).read_text());meta=asset_metadata(artifact)
    if meta['payloadBytes']>32768:raise ValueError('Model exceeds32KiB')
    if record.get('selectionFrozen') is not True or any(record.get(k)!=meta[k] for k in ['payloadSha256','manifestSha256']):raise ValueError('Exact frozen hashes required')
    with Path(receipt).open('x') as f:json.dump({'selection':str(selection),'selectionSha256':hashlib.sha256(Path(selection).read_bytes()).hexdigest(),'candidate':meta,'accessedAt':datetime.now(timezone.utc).isoformat()},f,indent=2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--partition',choices=['dev','holdout'],default='dev');p.add_argument('--selection');p.add_argument('--output',default='eval/typo-evaluation.json');a=p.parse_args()
    if a.partition=='holdout':
        if not a.selection:raise ValueError('Selection required')
        claim(a.selection,a.candidate,'eval/typo-holdout-access.json')
    source=f'data/typo/{a.partition}.jsonl';rows=read_rows(source)
    baseline=evaluate(BASELINE,rows);candidate=evaluate(a.candidate,rows)
    labels=sorted({r['cleanLabel'] for r in rows});delta=np.array([c['hit']-b['hit'] for b,c in zip(baseline['perQuery'],candidate['perQuery'])]);groups=[np.array([i for i,r in enumerate(rows) if r['cleanLabel']==label]) for label in labels]
    rng=np.random.default_rng(290917);samples=[float(delta[np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])].mean()) for _ in range(3000)]
    report={'scope':'Synthetic one-edit spelling; no lexical ranking, inference rewriting or runtime dictionary. New augmentation clean labels held apart; baseline pretraining may have seen them. Not fresh semantic evidence.','partition':source,'sourceSha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'baselineArtifact':asset_metadata(BASELINE),'candidateArtifact':asset_metadata(a.candidate),'baseline':baseline,'candidate':candidate,'pairedTop1Delta':float(delta.mean()),'cleanLabelClusterBootstrap95':np.quantile(samples,[.025,.975]).tolist()}
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['baseline','candidate']},indent=2));print('top1',baseline['top1'],candidate['top1'])
if __name__=='__main__':main()
