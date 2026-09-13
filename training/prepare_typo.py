"""Synthetic typo consistency pairs split by clean label before augmentation.

Only existing TRAIN labels are inputs. These are spelling diagnostics, not reviewed
semantic relevance judgments or a fresh evaluation of the original model.
"""
import hashlib,json,random
from collections import defaultdict
from pathlib import Path
from training.features import normalize

def corruptions(text):
    result=set()
    for i,c in enumerate(text):
        if not c.isascii() or not c.isalpha():continue
        # One edit; retain at least three letters in the affected token.
        start=text.rfind(' ',0,i)+1;end=text.find(' ',i);end=len(text) if end<0 else end
        if end-start<4:continue
        result.add(text[:i]+text[i+1:])
        result.add(text[:i]+c+text[i:])
        if i+1<len(text) and text[i+1].isalpha() and c!=text[i+1]:result.add(text[:i]+text[i+1]+c+text[i+2:])
        keyboard='qwertyuiopasdfghjklzxcvbnm';j=keyboard.find(c)
        if j>=0:result.add(text[:i]+keyboard[(j+1)%len(keyboard)]+text[i+1:])
    return sorted(result-{text})

def main():
    sources=['data/expanded/train.jsonl','data/navigation-splits/train.jsonl']
    labels=sorted({normalize(c['label']) for p in sources for line in Path(p).read_text().splitlines() for c in json.loads(line)['candidates'] if 4<=len(normalize(c['label']))<=64})
    buckets={'train':[],'dev':[],'holdout':[],'diagnostic':[]}
    for label in labels:
        bucket=int(hashlib.sha256(label.encode()).hexdigest()[:8],16)%10
        split='diagnostic' if label in ['profile','profiles'] else 'train' if bucket<7 else 'dev' if bucket<9 else 'holdout'
        buckets[split].append(label)
    allclean=set(labels);owners=defaultdict(set)
    variants={label:corruptions(label) for label in labels}
    for label,items in variants.items():
        for text in items:owners[text].add(label)
    out=Path('data/typo');out.mkdir(exist_ok=False)
    report={'scope':'Synthetic one-edit spelling diagnostics from original TRAIN labels. Clean labels disjoint across new augmentation splits, but may have been seen by pretrained baseline. Profile/Profiles excluded from augmentation training and selection; known bug is a separate consumed diagnostic. Not human judgments.','sources':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},'partitions':{}}
    for split,clean in buckets.items():
        pairs=[];rows=[]
        for label in clean:
            allowed=[t for t in variants[label] if t not in allclean and len(owners[t])==1]
            rng=random.Random('typo-v1:'+label);rng.shuffle(allowed)
            for typo in allowed[:12]:
                pairs.append([typo,label])
                # A deterministic 50-label menu including similar surface forms.
                def similarity(other):return len(set(label)&set(other))/max(1,len(set(label)|set(other)))
                distractors=sorted([x for x in labels if x!=label],key=lambda x:(-similarity(x),x))[:49]
                menu=[label,*distractors];rng.shuffle(menu)
                rows.append({'id':f'{split}-{len(rows):05d}','query':typo,'cleanLabel':label,'candidates':[{'id':x,'label':x} for x in menu],'relevance':{x:3 if x==label else 0 for x in menu},'judgment_status':'synthetic unique one-edit origin, not reviewed semantic negatives'})
        if split=='train':
            path=out/'train-pairs.json';path.write_text(json.dumps(pairs,indent=2)+'\n')
        else:
            path=out/f'{split}.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        report['partitions'][split]={'cleanLabels':clean,'pairs':len(pairs),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:(len(v['cleanLabels']),v['pairs']) for k,v in report['partitions'].items()})
if __name__=='__main__':main()
