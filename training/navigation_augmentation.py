"""Fixed train-only removal of one polite leading prefix; no semantic rewriting."""
from training.features import normalize,tokens

PREFIXES=('can you','could you','how do i','i want to','i would like to','please','show me','tell me')

def shorten_query(query,prefixes=PREFIXES):
    text=normalize(query)
    for prefix in prefixes:
        if text.startswith(prefix+' '):
            remainder=text[len(prefix)+1:]
            return remainder if len(tokens(remainder))>=2 else None
    return None

def augment_positive_rows(rows,prefixes=PREFIXES):
    augmented=[]
    for row in rows:
        if not any(grade>=2 for grade in row['relevance'].values()):continue
        text=shorten_query(row['query'],prefixes)
        if text:
            augmented.append({**row,'id':row['id']+'-polite-prefix-removed','query':text,'augmentation_parent':row['id'],'augmentation':'one-fixed-polite-prefix-removed'})
    return augmented

def documented_positive_pairs(rows):
    pairs=[]
    for row in rows:
        labels={c['id']:c['label'] for c in row['candidates']}
        for cid,grade in row['relevance'].items():
            if grade>=2:
                if cid not in labels:raise ValueError('positive not present in menu')
                pairs.append((row['query'],labels[cid]))
    if not pairs:raise ValueError('no documented positive pairs')
    return pairs
