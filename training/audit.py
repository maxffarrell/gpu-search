"""Recompute provenance checksums and hash-bucket collision accounting."""
import json,hashlib
from pathlib import Path
from training.features import tokens,hash_bytes

def main():
    rows=[json.loads(l) for l in Path('data/train.jsonl').read_text().splitlines()]
    texts={r['query'] for r in rows}|{c['label'] for r in rows for c in r['candidates']}
    families={'word':set(),'char':set()}
    for text in texts:
        chars_left=512
        for token in tokens(text)[:32]:
            token=token[:64];families['word'].add(('w:'+token).encode());elems=[b'\x01']+[b'\x00'+ord(c).to_bytes(4,'little') for c in token]+[b'\x02']
            for n in (2,3,4):
                for i in range(len(elems)-n+1):
                    if chars_left>0:families['char'].add(b'c:'+b''.join(elems[i:i+n]));chars_left-=1
    audit={}
    for name,values in families.items():
        buckets={}
        for value in values:buckets.setdefault(hash_bytes(value),[]).append(value)
        audit[name]={'uniqueFeaturePreimages':len(values),'occupiedBuckets':len(buckets),'collidingDistinctFeaturesBeyondFirst':len(values)-len(buckets),'bucketsWithMultipleFeatures':sum(len(v)>1 for v in buckets.values())}
    Path('eval/hash-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    paths=[*Path('runs').glob('*/weights.npz'),*Path('data').glob('*.jsonl'),Path('packages/model/experimental/weights.bin')]
    Path('eval/artifact-hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths) if p.exists()},indent=2)+'\n')
    print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
