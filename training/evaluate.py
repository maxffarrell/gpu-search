import argparse,json,math,re
from pathlib import Path
import numpy as np
from training.features import normalize,tokens
from training.model import encode_numpy,candidate_vectors

def osa(a,b):
    d=[[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):d[i][0]=i
    for j in range(len(b)+1):d[0][j]=j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]!=b[j-1]))
            if i>1 and j>1 and a[i-1]==b[j-2] and a[i-2]==b[j-1]:d[i][j]=min(d[i][j],d[i-2][j-2]+1)
    return d[-1][-1]
def lexical(q,c,aliases=None):
    original_query=q
    q=normalize(q)
    if not q:return (0,0,0)
    if q==normalize(c['label']):return(3,0,1)
    best=(0,0,0)
    for index,field in enumerate([c['label'],*c.get('aliases',[]),*(aliases or {}).get(c['label'],[])]):
        f=normalize(field);score=min(1,len(q.replace(' ',''))/max(1,len(f.replace(' ',''))))
        if index and q==f:rank=(2,5,1)
        elif len(q)>=2 and f.startswith(q):rank=(2,4,score)
        else:
            qt=tokens(original_query);ft=tokens(field);cursor=0
            for token in ft:
                if cursor<len(qt) and token.startswith(qt[cursor]):cursor+=1
            if qt and cursor==len(qt) and len(q.replace(' ',''))>=3:rank=(2,3,score)
            elif 2<=len(q)<=6 and len(ft)>=2 and q==''.join(t[0] for t in ft):rank=(2,2,1)
            else:
                L=max(len(q),len(f));distance=osa(q,f)
                rank=(2,1,1-distance/L) if min(len(q),len(f))>=4 and distance<=(1 if L<=7 else 2) and distance/L<=.25 else (0,0,0)
        best=max(best,rank)
    return best

def rank(row,scores=None,cutoff=1.01,aliases=None):
    pairs=[]
    for i,c in enumerate(row['candidates']):
        k=lexical(row['query'],c,aliases)
        if k[0]:pairs.append((k,i))
        elif scores is not None and len(normalize(row['query']))>=3 and scores[i]>=cutoff:pairs.append(((1,0,float(scores[i])),i))
    return [i for _,i in sorted(pairs,key=lambda x:(*[-v for v in x[0]],x[1]))]
def ndcg(row,indices):
    grades=[row['relevance'].get(c['id'],0) for c in row['candidates']]
    def dcg(gs):return sum((2**g-1)/math.log2(i+2) for i,g in enumerate(gs[:5]))
    ideal=dcg(sorted(grades,reverse=True));return dcg([grades[i] for i in indices])/ideal if ideal else 0.
def summarize(rows,rankings):
    relevant=[i for i,r in enumerate(rows) if max(r['relevance'].values())>=2];none=[i for i,r in enumerate(rows) if r['query_kind']=='no-match']
    result={'count':len(rows),'relevantCount':len(relevant),'noMatchCount':len(none),'ndcg5':float(np.mean([ndcg(rows[i],rankings[i]) for i in relevant])) if relevant else None,'coverage':float(np.mean([bool(rankings[i]) for i in relevant])) if relevant else None,'noMatchAnyRate':float(np.mean([bool(rankings[i]) for i in none])) if none else None}
    result['noMatchSemanticRate']=float(np.mean([any(not lexical(rows[i]['query'],rows[i]['candidates'][j])[0] for j in rankings[i]) for i in none])) if none else None
    result['ndcgPolicy']='Queries with at least one grade>=2; current synthetic corpus uses grades 0 and 3 only'
    for k in [1,3,5]:
        result[f'recall{k}']=float(np.mean([sum(rows[i]['relevance'][rows[i]['candidates'][j]['id']]>=2 for j in rankings[i][:k])/sum(g>=2 for g in rows[i]['relevance'].values()) for i in relevant])) if relevant else None
    result['mrr']=float(np.mean([next((1/(n+1) for n,j in enumerate(rankings[i]) if rows[i]['relevance'][rows[i]['candidates'][j]['id']]>=2),0) for i in relevant])) if relevant else None
    return result

def evaluate(checkpoint,split='dev'):
    rows=[json.loads(l) for l in Path(f'data/{split}.jsonl').read_text().splitlines()];p=dict(np.load(Path(checkpoint)/'weights.npz'));meta=json.loads((Path(checkpoint)/'training.json').read_text());aliases=json.loads(Path('data/aliases.json').read_text())
    scores=[candidate_vectors(p,r['candidates'],meta['family'])@encode_numpy(p,[r['query']],meta['family'])[0] for r in rows]
    none=[float(max(s)) for r,s in zip(rows,scores) if r['query_kind']=='no-match']
    # With 20 synthetic no-match development records, allow at most one false positive.
    cutoff=float(np.nextafter(np.float32(sorted(none,reverse=True)[min(len(none)-1,int(.05*len(none)))]),np.float32(np.inf))) if none else 1.01
    systems={'lexical':[rank(r) for r in rows],'lexical+train-aliases':[rank(r,aliases=aliases) for r in rows],'learned-before-cutoff':[rank(r,s,-1) for r,s in zip(rows,scores)],'learned-after-cutoff':[rank(r,s,cutoff) for r,s in zip(rows,scores)]}
    semantic=[i for i,r in enumerate(rows) if r['query_kind']=='semantic-only' and not any(lexical(r['query'],c)[0] and r['relevance'][c['id']]>=2 for c in r['candidates'])]
    base=systems['lexical+train-aliases'];model=systems['learned-after-cutoff'];delta=np.array([ndcg(rows[i],model[i])-ndcg(rows[i],base[i]) for i in semantic]);rng=np.random.default_rng(20260912)
    ci=np.quantile(rng.choice(delta,(10000,len(delta)),replace=True).mean(axis=1),[.025,.975]).tolist()
    return {'checkpoint':str(checkpoint),'split':split,'cutoff':cutoff,'cutoffSelectedOn':split,'warning':'Synthetic unreviewed development diagnostics, not final release evidence; calibration and scoring reuse development','semanticCount':len(semantic),'semanticDelta':float(delta.mean()),'pairedBootstrap95':ci,'systems':{name:{'overall':summarize(rows,rankings),'semanticOnly':summarize([rows[i] for i in semantic],[rankings[i] for i in semantic])} for name,rankings in systems.items()},'perQuery':[{'id':r['id'],'query':r['query'],'kind':r['query_kind'],'scores':[float(x) for x in s],'rankings':{k:v[i] for k,v in systems.items()}} for i,(r,s) in enumerate(zip(rows,scores))]}
def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev',choices=['train','dev']);p.add_argument('--checkpoint',default='runs/selected');a=p.parse_args()
    if not Path(a.checkpoint).exists():p.error('No release checkpoint selected: use an explicit experimental runs/<name> directory; gate not passed')
    result=evaluate(a.checkpoint,a.split);out=Path(a.checkpoint)/f'{a.split}-evaluation.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='perQuery'},indent=2))
if __name__=='__main__':main()
