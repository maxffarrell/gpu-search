"""Independent NumPy feature ablation audit. Does not import MLX experiment encoder."""
import hashlib,json,sys,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from training.features import features,tokens,hash_bytes
from training.evaluate_expanded import read_rows,calibrate,rankings,summarize,bootstrap
from training.evaluate import lexical

cal=read_rows('data/expanded/calibration.jsonl');dev=read_rows('data/expanded/dev.jsonl');all_rows=cal+dev
texts=sorted({t for row in all_rows for t in [row['query'],*[c['label'] for c in row['candidates']]]})
base={t:features(t) for t in texts}
bigrams={}
for text in texts:
 ts=[t[:64] for t in tokens(text)[:32]];ids=[]
 for a,b in zip(ts,ts[1:]):
  a,b=a.encode(),b.encode();ids.append(hash_bytes(b'b:'+len(a).to_bytes(4,'little')+a+len(b).to_bytes(4,'little')+b))
 bigrams[text]=ids
lex=[[lexical(row['query'],c) for c in row['candidates']] for row in dev]
semantic=[i for i,row in enumerate(dev) if any(v>=2 for v in row['relevance'].values()) and not any(t[0] and row['relevance'].get(c['id'],0)>=2 for t,c in zip(lex[i],row['candidates']))]

def scores(params,arm):
 vs={}
 for text,f in base.items():
  w=params['word'][f['wordIds']].mean(axis=0) if f['wordIds'] else np.zeros(16,np.float32)
  c=params['char'][f['charIds']].mean(axis=0) if f['charIds'] else np.zeros(16,np.float32)
  if arm=='bigram25' and bigrams[text]:w=.75*w+.25*params['word'][bigrams[text]].mean(axis=0)
  z=(.75*w+.25*c) if arm=='word75' else .5*(w+c)
  norm=np.linalg.norm(z);vs[text]=z/max(norm,1e-8)
 return [[float(vs[row['query']]@vs[c['label']]) for c in row['candidates']] for row in all_rows]

def measure(sc,cutoff=None):
 cutoff=calibrate(cal,sc[:len(cal)]) if cutoff is None else cutoff
 ranks=rankings(dev,sc[len(cal):],cutoff,lex)
 result={'cutoff':cutoff,'overall':summarize(dev,ranks,lex),'semantic':summarize([dev[i] for i in semantic],[ranks[i] for i in semantic],[lex[i] for i in semantic]),'slices':{}}
 for source in sorted({r['source_id'] for r in dev}):
  ids=[i for i,r in enumerate(dev) if r['source_id']==source];result['slices'][source]=summarize([dev[i] for i in ids],[ranks[i] for i in ids],[lex[i] for i in ids])
 for name in sorted({r['evaluation_slice'] for r in dev}):
  ids=[i for i,r in enumerate(dev) if r['evaluation_slice']==name];result['slices'][name]=summarize([dev[i] for i in ids],[ranks[i] for i in ids],[lex[i] for i in ids])
 return result,ranks

systems={};ranks={}
for folder in sorted(Path('runs').glob('features-*-seed*-e10')):
 meta=json.loads((folder/'training.json').read_text());arm=meta['arm'];seed=meta['seed']
 for stage in ['selected','final']:
  file=folder/('weights.npz' if stage=='selected' else 'final-weights.npz')
  if not file.exists():continue
  name=f'{arm}-seed{seed}-{stage}';sc=scores(dict(np.load(file)),arm);report,ranks[name]=measure(sc);systems[name]=report
  report['checkpoint']=str(file);report['weightHash']=hashlib.sha256(file.read_bytes()).hexdigest()
 if 'artifact' in meta:
  artifact=Path(meta['artifact']);manifest=json.loads((artifact/'manifest.json').read_text());payload=(artifact/'weights.bin').read_bytes();params={}
  for tensor in manifest['tensors']:
   lo=tensor['byteOffset'];params[tensor['name']]=np.frombuffer(payload[lo:lo+tensor['byteLength']],np.int8).reshape(tensor['shape']).astype(np.float32)*np.float32(tensor['scale'])
  sc=scores(params,arm);name=f'{arm}-seed{seed}-int8';report,ranks[name]=measure(sc);systems[name]=report
  report['payloadBytes']=len(payload);report['gzipWeightBytes']=len(zlib.compress(payload,9,wbits=31));report['scopeOfBytes']='Weights only; not complete browser transfer budget.'
  selected=systems[f'{arm}-seed{seed}-selected'];fixed,_=measure(sc,selected['cutoff']);report['fixedFloatCutoffSemanticNdcgLoss']=selected['semantic']['ndcg5']-fixed['semantic']['ndcg5'];report['recalibratedSemanticNdcgLoss']=selected['semantic']['ndcg5']-report['semantic']['ndcg5']
  report['maxScoreDifferenceFromFloat']=max(abs(x-y) for xs,ys in zip(sc,scores(dict(np.load(folder/'weights.npz')),arm)) for x,y in zip(xs,ys))
comparisons={}
for seed in [17,29,43]:
 for stage in ['selected','final']:
  a=f'bigram25-seed{seed}-{stage}';b=f'control-seed{seed}-{stage}'
  if a in ranks and b in ranks:comparisons[f'seed{seed}-{stage}']=bootstrap(dev,ranks[a],ranks[b],semantic)
report={'scope':'Independent NumPy on expanded dev/calibration only; no final test and no browser parity','systems':systems,'pairedSemanticComparisons':comparisons,'interpretation':'Selected epochs are constrained by dev5% no-match; equal-update final epochs shown even when unsafe. Confidence intervals condition on selected checkpoints and do not undo development selection bias.'}
Path('eval/feature-experiments-independent.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'comparisons':comparisons,'int8':{n:{'ndcg':v['semantic']['ndcg5'],'noMatch':v['overall']['noMatchSemanticRate'],'ui':v['slices']['vscode-settings']['ndcg5'],'loss':v.get('recalibratedSemanticNdcgLoss')} for n,v in systems.items() if n.endswith('int8')}},indent=2))
