"""Aggregate development diagnostics without opening the reserved synthetic test set."""
import json,platform,subprocess
from pathlib import Path
import numpy as np
from training.evaluate import evaluate,rank,summarize,lexical
from training.features import normalize

def tfidf(rows):
    # Fit IDF on train candidate labels only. No dev/test labels or queries in fit.
    train=[json.loads(l) for l in Path('data/train.jsonl').read_text().splitlines()]
    texts=sorted({c['label'] for r in train for c in r['candidates']})
    def grams(text):
        t='^'+normalize(text)+'$';return [t[i:i+n] for n in (2,3,4) for i in range(max(0,len(t)-n+1))]
    counts={}
    for text in texts:
        for g in set(grams(text)):counts[g]=counts.get(g,0)+1
    def vec(text):
        v={}
        for g in grams(text):v[g]=v.get(g,0)+1
        v={g:n*(np.log((1+len(texts))/(1+counts.get(g,0)))+1) for g,n in v.items()};norm=np.sqrt(sum(x*x for x in v.values()))
        return {g:x/norm for g,x in v.items()} if norm else {}
    def score(q,c):
        a,b=vec(q),vec(c);return sum(x*b.get(g,0) for g,x in a.items())
    scores=[[score(r['query'],c['label']) for c in r['candidates']] for r in rows]
    none=sorted([max(s) for r,s in zip(rows,scores) if r['query_kind']=='no-match'],reverse=True)
    cutoff=float(np.nextafter(none[int(.05*len(none))],np.inf))
    return scores,cutoff

def main():
    names=json.loads(Path('eval/experiments.json').read_text());table=[]
    for name in names:
        result=evaluate('runs/'+name);Path('runs',name,'dev-evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
        meta=json.loads(Path('runs',name,'training.json').read_text());after=result['systems']['learned-after-cutoff']
        table.append({'name':name,'epochs':meta['epochs'],'seconds':meta['trainingSeconds'],'semanticBefore':result['systems']['learned-before-cutoff']['semanticOnly']['ndcg5'],'semanticAfter':after['semanticOnly']['ndcg5'],'semanticCoverage':after['semanticOnly']['coverage'],'overall':after['overall']['ndcg5'],'noMatchAnyRate':after['overall']['noMatchAnyRate'],'cutoff':result['cutoff'],'delta':result['semanticDelta'],'bootstrap95':result['pairedBootstrap95'],'provisionalNumericGate':result['semanticDelta']>=.05 and result['pairedBootstrap95'][0]>0})
    rows=[json.loads(l) for l in Path('data/dev.jsonl').read_text().splitlines()];aliases=json.loads(Path('data/aliases.json').read_text())
    baseline=[{'id':r['id'],'query':r['query'],'candidates':r['candidates'],'lexicalIds':[r['candidates'][i]['id'] for i in rank(r)],'aliasIds':[r['candidates'][i]['id'] for i in rank(r,aliases=aliases)]} for r in rows]

    for query,label in [('fooBar','Foo Bar'),('foo-bar','fooBar')]:
        r={'query':query,'candidates':[{'id':'camel','label':label}]}
        baseline.append({'id':'adversarial-'+query,**r,'lexicalIds':[r['candidates'][i]['id'] for i in rank(r)],'aliasIds':[r['candidates'][i]['id'] for i in rank(r,aliases=aliases)]})
    Path('eval/baseline-fixtures.json').write_text(json.dumps(baseline,indent=2)+'\n')
    scores,cutoff=tfidf(rows);tf_ranks=[rank(r,s,cutoff) for r,s in zip(rows,scores)];sem=[i for i,r in enumerate(rows) if r['query_kind']=='semantic-only']
    summary={'status':'STOP: no release model selected; unreviewed synthetic data cannot establish release quality','device':json.loads(Path('eval/environment.json').read_text())['device'],'platform':platform.platform(),'mlx':'0.31.1','seeds':[17,29,43],'trainCount':120,'devCount':80,'devSemanticCount':36,'devNoMatchCount':20,'reservedSyntheticTestCount':40,'testOpenedForScoring':False,'humanReviewedCount':0,'baseline':evaluate('runs/'+names[0])['systems']['lexical+train-aliases'],'charTfidf':{'fit':'train candidate labels only; unseen grams use smoothed zero document frequency; original implementation','cutoff':cutoff,'overall':summarize(rows,tf_ranks),'semanticOnly':summarize([rows[i] for i in sem],[tf_ranks[i] for i in sem])},'experiments':table,'unrun':['general embedding reference (no pinned local teacher)','human reviewed held-out evaluation','distillation','QAT/int6 architecture optimization stopped at feasibility gate','real product-source data curation','alias/context composition ablations (dataset contains labels only)'],'limitations':['Small synthetic menus, one positive each; not representative of real menus or multi-positive judgment quality','Development used for early stopping, cutoff and diagnostics; confidence intervals are descriptive after model selection','Train alias dictionary has no covered dev labels, so lexical+alias equals lexical on these disjoint domains','No-match strings are synthetic and easy; a 5% development rate is not evidence of calibrated deployment behavior','Projected model failures and low post-cutoff coverage suggest inadequate data, not proof architecture cannot work']}
    Path('eval/feasibility.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'status':summary['status'],'models':table,'tfidf':summary['charTfidf']},indent=2))
if __name__=='__main__': main()
