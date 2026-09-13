"""Read-only diagnostics. The consumed synthetic test is now regression data, not fresh evidence."""
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from training.features import tokens,hash_bytes,normalize
from training.evaluate_release_candidate import model_scores
from training.evaluate_expanded import Evaluator,read_rows,calibrate


def quantiles(values):
    return {name:float(value) for name,value in zip(['min','p10','median','p90','max'],np.quantile(values,[0,.1,.5,.9,1]))} if values else {}


def main():
    paths={'train':'data/expanded/train.jsonl','calibration':'data/expanded/calibration.jsonl','expandedDev':'data/expanded/dev.jsonl','originalDev':'data/dev.jsonl','consumedSyntheticRegression':'data/test.jsonl'}
    rows={key:read_rows(path) for key,path in paths.items()}
    train_texts={t for r in rows['train'] for t in [r['query'],*[c['label'] for c in r['candidates']]]}
    train_words={w for t in train_texts for w in tokens(t)[:32]}
    train_labels={normalize(c['label']) for r in rows['train'] for c in r['candidates'] if r['relevance'].get(c['id'],0)>=2}
    distribution={};models={}
    for name,part in rows.items():
        positive=[r for r in part if any(v>=2 for v in r['relevance'].values())]
        target_labels=[c['label'] for r in positive for c in r['candidates'] if r['relevance'].get(c['id'],0)>=2]
        words=[w for r in part for w in tokens(r['query'])]
        distribution[name]={'count':len(part),'queryTokens':quantiles([len(tokens(r['query'])) for r in part]),'queryScalars':quantiles([len(r['query']) for r in part]),'candidateCounts':dict(Counter(len(r['candidates']) for r in part)),'positiveCount':len(positive),'positiveTargetsSeenInTraining':sum(normalize(t) in train_labels for t in target_labels),'queryUnseenTokenOccurrences':sum(w not in train_words for w in words),'queryTokenOccurrences':len(words),'noMatchCount':sum(not any(r['relevance'].values()) for r in part),'removedPositiveNoMatchCount':sum(r.get('negative_status')=='removed-positive-uncertain' for r in part)}
    for name,directory in [('current','packages/model/candidate'),('rejectedBigram','packages/model/candidate-v2')]:
        cs=model_scores(directory,rows['calibration']);cutoff=calibrate(rows['calibration'],cs)
        data={}
        for split in ['calibration','expandedDev','originalDev','consumedSyntheticRegression']:
            part=rows[split];scores=cs if split=='calibration' else model_scores(directory,part)
            semantic=Evaluator([],part).semantic;relevant_scores=[];top_correct=0;top_correct_rejected=0;max_scores=[];examples=[]
            for i in semantic:
                row=part[i];values=scores[i];best=int(np.argmax(values));rel=[j for j,c in enumerate(row['candidates']) if row['relevance'].get(c['id'],0)>=2]
                relevant_scores.append(max(values[j] for j in rel));max_scores.append(max(values))
                correct=best in rel;top_correct+=correct;top_correct_rejected+=correct and values[best]<cutoff
                if split=='consumedSyntheticRegression':examples.append({'query':row['query'],'target':[row['candidates'][j]['label'] for j in rel],'bestLabel':row['candidates'][best]['label'],'bestScore':values[best],'bestRelevantScore':relevant_scores[-1],'top1Correct':correct,'cutoff':cutoff})
            none_max=[max(v) for r,v in zip(part,scores) if not any(r['relevance'].values())]
            data[split]={'semanticCount':len(semantic),'semanticQueryTokens':quantiles([len(tokens(part[i]['query'])) for i in semantic]),'bestRelevantCosine':quantiles(relevant_scores),'maxMenuCosine':quantiles(max_scores),'noMatchMaxCosine':quantiles(none_max),'rawTop1Correct':int(top_correct),'correctRawTop1SuppressedByCutoff':int(top_correct_rejected),'relevantAboveCutoff':sum(v>=cutoff for v in relevant_scores),'examples':examples}
        manifest=json.loads(Path(directory,'manifest.json').read_text());models[name]={'path':directory,'sha256':manifest['payloadSha256'],'cutoff':cutoff,'partitions':data}
    # Flags identify judgment-review work, not automatically corrected labels.
    labels={c['id']:c['label'] for part in rows.values() for r in part for c in r['candidates']}
    risks={}
    for split in ['train','calibration','expandedDev']:
        flagged=[];missing=0;exact_contradictions=0;multi=0
        for row in rows[split]:
            by_label=defaultdict(set)
            for c in row['candidates']:by_label[normalize(c['label'])].add(row['relevance'].get(c['id'],-1))
            exact_contradictions+=any(any(g>=2 for g in gs) and 0 in gs for gs in by_label.values())
            multi+=sum(g>=2 for g in row['relevance'].values())>1
            if row.get('negative_status')!='removed-positive-uncertain':continue
            target=labels.get(f"{row['source_id']}:{row['source_intent']}")
            if not target:missing+=1;continue
            a=set(tokens(target));best=(0,None)
            for c in row['candidates']:
                b=set(tokens(c['label']));shared=len(a&b);score=shared/max(1,len(a|b))
                if shared>=2 and score>best[0]:best=(score,c['label'])
            if best[0]>=.5:flagged.append({'query':row['query'],'removedTarget':target,'remainingCandidate':best[1],'tokenJaccard':best[0]})
        risks[split]={'removedPositiveWithRelatedCandidateFlags':len(flagged),'rule':'at least2 shared label tokens and token Jaccard>=0.5; review flag, not proof of wrong label','examples':sorted(flagged,key=lambda r:-r['tokenJaccard'])[:12],'targetMappingMissing':missing,'exactLabelContradictoryMenus':exact_contradictions,'multiPositiveMenus':multi}
    tables={name:defaultdict(set) for name in ['word','char','wordPlusBigram']}
    for text in train_texts:
        ts=[t[:64] for t in tokens(text)[:32]];char_count=0
        for token in ts:
            word=('w:'+token).encode();tables['word'][hash_bytes(word)].add(word);tables['wordPlusBigram'][hash_bytes(word)].add(word)
            elements=[b'\x01']+[b'\x00'+ord(c).to_bytes(4,'little') for c in token]+[b'\x02']
            for n in [2,3,4]:
                for i in range(len(elements)-n+1):
                    if char_count<512:
                        value=b'c:'+b''.join(elements[i:i+n]);tables['char'][hash_bytes(value)].add(value)
                    char_count+=1
        for a,b in zip(ts,ts[1:]):
            a=a.encode();b=b.encode();value=b'b:'+len(a).to_bytes(4,'little')+a+len(b).to_bytes(4,'little')+b
            tables['wordPlusBigram'][hash_bytes(value)].add(value)
    collision={name:{'uniqueFeaturePreimages':sum(map(len,table.values())),'occupiedBuckets':len(table),'multiPreimageBuckets':sum(len(v)>1 for v in table.values()),'largestBucket':max(map(len,table.values())),'featuresBeyondFirstPerBucket':sum(len(v)-1 for v in table.values())} for name,table in tables.items()}
    result={'scope':'Read-only failure analysis; consumed test is development regression only; no weights fit or thresholds changed','distribution':distribution,'fixedModels':models,'supervisionRisks':risks,'trainingFeatureCollisions':collision,'sourceSha256':{path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths.values()},'limits':'Unreviewed source-class labels; correlation and collisions are diagnostic, not causal ablation evidence'}
    Path('eval/readiness-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'distribution':distribution,'risks':{k:{n:v for n,v in r.items() if n!='examples'} for k,r in risks.items()},'collisions':collision},indent=2))


if __name__=='__main__':main()
