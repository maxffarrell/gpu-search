"""Isolated same-byte pooled16 feature ablations; no deployment contract mutation."""
import argparse, hashlib, json, platform, time
from pathlib import Path
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from training.features import features, tokens, hash_bytes
from training.train_expanded import SparseCache, Menus, Development, read, validate_rows

CONTRACTS={
 'control':{'wordMix':1.,'bigramMix':0.,'wordFamily':.5,'charFamily':.5},
 'word75':{'wordMix':1.,'bigramMix':0.,'wordFamily':.75,'charFamily':.25},
 'bigram25':{'wordMix':.75,'bigramMix':.25,'wordFamily':.5,'charFamily':.5},
}
class FeatureCache(SparseCache):
 def __init__(self,texts,arm):
  self.texts=list(dict.fromkeys(texts));self.lookup={t:i for i,t in enumerate(self.texts)};self.arm=arm
  fs=[features(t) for t in self.texts];self.truncated=sum(f['truncated'] for f in fs)
  config=CONTRACTS[arm];wi=[];ww=[];ci=[];cw=[]
  for text,f in zip(self.texts,fs):
   ws=f['wordIds'];cs=f['charIds'];ts=[t[:64] for t in tokens(text)[:32]]
   # Length-prefixed UTF8 tokens make literal user punctuation unambiguous.
   bg=[]
   if config['bigramMix']:
    for a,b in zip(ts,ts[1:]):
     ab,bb=a.encode(),b.encode();bg.append(hash_bytes(b'b:'+len(ab).to_bytes(4,'little')+ab+len(bb).to_bytes(4,'little')+bb))
   # Single tokens retain unit word weight; no missing-bigram scale penalty.
   bm=config['bigramMix'] if bg else 0.
   wi.append(ws+bg);ww.append([(1-bm)*config['wordFamily']/len(ws)]*len(ws) if ws else [])
   if bg:ww[-1]+=[bm*config['wordFamily']/len(bg)]*len(bg)
   ci.append(cs);cw.append([config['charFamily']/len(cs)]*len(cs) if cs else [])
  self.arrays=[]
  for ids,weights in [(wi,ww),(ci,cw)]:
   width=max(1,max(map(len,ids),default=0));ii=np.zeros((len(ids),width),np.int32);vv=np.zeros((len(ids),width),np.float32)
   for n,(row,value) in enumerate(zip(ids,weights)):ii[n,:len(row)]=row;vv[n,:len(row)]=value
   self.arrays.extend([mx.array(ii),mx.array(vv)])
  mx.eval(*self.arrays)
 def encode(self,params,ids):
  wi,ww,ci,cw=self.arrays
  v=mx.sum(params['word'][wi[ids]]*ww[ids,:,None],axis=1)+mx.sum(params['char'][ci[ids]]*cw[ids,:,None],axis=1)
  return v/mx.maximum(mx.sqrt(mx.sum(v*v,axis=-1,keepdims=True)),1e-8)

def run(arm,seed,epochs,rows,development,hashes):
 folder=Path(f'runs/features-{arm}-seed{seed}-e{epochs}')
 if folder.exists():raise FileExistsError(folder)
 folder.mkdir(parents=True)
 supervised=[r for r in rows if any(g>=2 for g in r['relevance'].values())]
 texts=[t for r in supervised for t in [r['query'],*[c['label'] for c in r['candidates']]]]
 cache=FeatureCache(texts,arm);menus=Menus(supervised,cache)
 development.cache=FeatureCache(development.cache.texts,arm)
 mx.random.seed(seed);rng=np.random.default_rng(seed)
 params={'word':mx.random.normal((1024,16))*.1,'char':mx.random.normal((1024,16))*.1}
 optimizer=optim.AdamW(learning_rate=.003,weight_decay=.0001)
 def loss(p,unique,qi,ci,pos,eligible):
  v=cache.encode(p,unique);score=mx.sum(v[qi,None,:]*v[ci],axis=-1)/.15
  return mx.mean(mx.logsumexp(mx.where(eligible,score,-1e9),axis=1)-mx.logsumexp(mx.where(pos,score,-1e9),axis=1))
 grad=mx.value_and_grad(loss);history=[];best=None;best_key=(-1.,-1.);start=time.perf_counter()
 for epoch in range(1,epochs+1):
  order=rng.permutation(len(supervised));losses=[]
  for offset in range(0,len(order),128):
   value,g=grad(params,*menus.batch(order[offset:offset+128]));g,_=optim.clip_grad_norm(g,1.);params=optimizer.apply_gradients(g,params);mx.eval(value,params,optimizer.state);losses.append(float(value))
  metrics=development.measure(params);entry={'epoch':epoch,'loss':float(np.mean(losses)),**metrics};history.append(entry)
  # Dev false-positive safety is part of selection; threshold itself is cal only.
  eligible=metrics['devNoMatchSemanticRate']<=.05
  key=(metrics['devSemanticNdcg5'],metrics['devRelevantCoverage'])
  if eligible and key>best_key:
   best_key=key;best=entry;np.savez(folder/'weights.npz',**{k:np.array(v) for k,v in params.items()})
  print(arm,seed,epoch,round(metrics['devSemanticNdcg5'],4),round(metrics['devNoMatchSemanticRate'],4),'eligible',eligible,flush=True)
 np.savez(folder/'final-weights.npz',**{k:np.array(v) for k,v in params.items()})
 meta={'arm':arm,'seed':seed,'epochs':epochs,'selected':best,'history':history,'featureContract':CONTRACTS[arm],'weightElements':32768,'int8WeightBytes':32768,'framework':'MLX0.31.1','device':str(mx.default_device()),'platform':platform.platform(),'trainingRows':len(rows),'positiveTrainingRows':len(supervised),'noMatchRowsExcluded':len(rows)-len(supervised),'optimizer':'AdamW lr.003 decay.0001 clip1 batch128 temperature.15','seconds':time.perf_counter()-start,'sourceHashes':hashes,'selection':'highest development post-cutoff semantic nDCG@5 then coverage among epochs with development no-match semantic<=.05; cutoff independently calibrated to<=.05 on calibration only','status':'weak-label feature experiment; no held-out test opened; no browser implementation for changed contracts'}
 (folder/'training.json').write_text(json.dumps(meta,indent=2)+'\n')
 if best:
  p=dict(np.load(folder/'weights.npz'));quant={};tensors=[];payload=b''
  for name,w in p.items():
   scale=np.float32(np.max(np.abs(w))/127) if np.any(w) else np.float32(1)
   codes=np.clip(np.sign(w)*np.floor(np.abs(w/scale)+.5),-127,127).astype(np.int8)
   quant[name]=(codes.astype(np.float32)*scale).astype(np.float32)
   tensors.append({'name':name,'shape':list(w.shape),'byteOffset':len(payload),'byteLength':codes.nbytes,'scale':float(scale),'bits':8});payload+=codes.tobytes()
  metrics=development.measure({k:mx.array(v) for k,v in quant.items()})
  export=Path(f'packages/model/experiments/features-{arm}-seed{seed}-e{epochs}');export.mkdir(parents=True)
  manifest={'formatVersion':'gpu-search-feature-experiment-v1','featureVersion':'gpu-search-features-v1' if arm=='control' else 'gpu-search-features-experiment-'+arm,'contract':CONTRACTS[arm],'bigramEncoding':'ASCII b: then uint32LE byte-length then UTF8 token A then uint32LE byte-length then UTF8 token B; FNV1a&1023; ordered adjacent bounded lexical tokens; shared word table','tensors':tensors,'payloadBytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'calibratedCutoff':metrics['cutoff'],'qualityValidated':False,'quantizedMetrics':metrics}
  (export/'weights.bin').write_bytes(payload);(export/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');meta['quantizedMetrics']=metrics;meta['artifact']=str(export)
  meta['quantizedSemanticNdcgLoss']=best['devSemanticNdcg5']-metrics['devSemanticNdcg5']
  (folder/'training.json').write_text(json.dumps(meta,indent=2)+'\n')
 return {k:v for k,v in meta.items() if k!='history'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',default=list(CONTRACTS),choices=list(CONTRACTS));p.add_argument('--seeds',nargs='+',type=int,default=[17]);p.add_argument('--epochs',type=int,default=10);a=p.parse_args()
 paths=[Path('data/expanded')/f'{split}.jsonl' for split in ['train','dev','calibration']];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 train,dev,cal=map(read,paths);validate_rows(train+dev+cal);development=Development(dev,cal)
 results=[]
 for seed in a.seeds:
  for arm in a.arms:results.append(run(arm,seed,a.epochs,train,development,hashes))
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
 output=Path('eval/feature-experiments-'+('-'.join(a.arms))+'-seeds'+('-'.join(map(str,a.seeds)))+'.json')
 output.write_text(json.dumps({'runs':results,'sourceHashes':hashes,'scope':'same16D 32768-parameter feature ablation; matched positive rows and optimizer updates; bounded10epoch screen, not final release evidence'},indent=2)+'\n')
 print('REPORT',output,flush=True)
if __name__=='__main__':main()
