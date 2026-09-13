"""Verify sparse MLX feature pooling against independent NumPy arithmetic."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
import numpy as np
from training.experiment_features import FeatureCache
from training.features import features,tokens,hash_bytes
texts=['','profile','my information','export project','project export','C++ C# C','👋 team','é e\u0301','foo foo foo','a'*300,' '.join(['repeat']*40)]
for line in Path('data/expanded/dev.jsonl').read_text().splitlines()[:128]:texts.append(json.loads(line)['query'])
reports=[]
for arm in ['control','bigram25']:
 path=Path(f'runs/features-{arm}-seed17-e10/weights.npz');params=dict(np.load(path));cache=FeatureCache(texts,arm);actual=cache.all_vectors({k:mx.array(v) for k,v in params.items()});expected=[]
 for text in cache.texts:
  f=features(text);w=params['word'][f['wordIds']].mean(0) if f['wordIds'] else np.zeros(16,np.float32);c=params['char'][f['charIds']].mean(0) if f['charIds'] else np.zeros(16,np.float32)
  ts=[t[:64] for t in tokens(text)[:32]];bs=[]
  for a,b in zip(ts,ts[1:]):
   aa,bb=a.encode(),b.encode();bs.append(hash_bytes(b'b:'+len(aa).to_bytes(4,'little')+aa+len(bb).to_bytes(4,'little')+bb))
  if arm=='bigram25' and bs:w=.75*w+.25*params['word'][bs].mean(0)
  z=.5*(w+c);expected.append(z/max(np.linalg.norm(z),1e-8))
 expected=np.array(expected);difference=float(np.max(np.abs(actual-expected)));score_difference=float(np.max(np.abs(actual@actual.T-expected@expected.T)))
 assert difference<=1e-4 and score_difference<=2e-4
 reports.append({'arm':arm,'texts':len(cache.texts),'maximumComponentError':difference,'maximumCosineError':score_difference,'passed':True})
Path('eval/feature-experiments-parity.json').write_text(json.dumps({'scope':'MLX sparse pooling vs independent NumPy; not browser parity','reports':reports},indent=2)+'\n')
print(json.dumps(reports,indent=2))
