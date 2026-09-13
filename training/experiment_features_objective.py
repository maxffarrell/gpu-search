"""Bounded interaction: winning feature scheme + no-match objective.

Runs existing objective trainer with explicitly scoped cache/output injection,
restored afterward. Does not alter baseline modules or inspect original transfer.
"""
import hashlib,json,struct
from pathlib import Path
import mlx.core as mx
import numpy as np
import yaml
import training.experiment_training as objective
from training.experiment_features import FeatureCache
from training.features_bigram import VERSION,features,encode_numpy
from training.export import quantize,pack
from training.train_expanded import read

class BigramCache(FeatureCache):
 def __init__(self,texts):super().__init__(texts,'bigram25')

def export(checkpoint,development):
 cp=Path(checkpoint);p=dict(np.load(cp/'weights.npz'));payload=b'';tensors=[];quant={}
 for name,w in p.items():
  q,scale=quantize(w,8);blob=pack(q,8);tensors.append({'name':name,'shape':list(w.shape),'elementCount':w.size,'bits':8,'scale':float(scale),'scaleF32LE':struct.pack('<f',scale).hex(),'byteOffset':len(payload),'byteLength':len(blob)});payload+=blob;quant[name]=q.astype(np.float32)*scale
 metrics=development.measure({k:mx.array(v) for k,v in quant.items()})
 out=Path('packages/model/experiments')/('features-bigram-no-match-'+cp.name.replace('experiment-no-match-',''))
 out.mkdir(parents=True)
 manifest={'formatVersion':'gpu-search-experimental-v1','featureVersion':VERSION,'normalizationVersion':'nfkc-ascii-v1','candidateCompositionVersion':'mean-alias-context025-v1','architecture':'pooled','dimension':16,'featureFamily':'both','modelId':out.name,'byteOrder':'little-endian','tensors':tensors,'payloadSha256':hashlib.sha256(payload).hexdigest(),'payloadBytes':len(payload),'validatedSemanticCutoff':None,'status':'EXPERIMENTAL; weak-label development selection; no final test opened','researchCalibrationCutoff':metrics['cutoff'],'quantizedMetrics':metrics,'checkpoint':str(cp)}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'weights.bin').write_bytes(payload)
 texts=['Profile','coworkers','my information','export project','project export','C++ C#','a a a','é e\u0301','👋 team','日本語 設定','a'*65,'🙂'*65,' '.join('token'+str(i) for i in range(35)),'repeat '*32,'C++ C# @user 123',''];stages=encode_numpy(quant,texts,stages=True)
 (out/'numerical-fixtures.json').write_text(json.dumps([{'text':text,'features':features(text),**{k:v[i].tolist() for k,v in stages.items()}} for i,text in enumerate(texts)],ensure_ascii=False,indent=2)+'\n')
 return {'artifact':str(out),'quantizedMetrics':metrics,'payloadSha256':manifest['payloadSha256']}

def main():
 cfg=yaml.safe_load(Path('configs/experiments-training.yaml').read_text());cfg['epochs']=30;cfg['tag']='bigram-no-match-e30';cfg['featureVersion']=VERSION
 paths=[Path('data/expanded')/f'{s}.jsonl' for s in ['train','dev','calibration']];cfg['sourceHashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 train,dev,cal=map(read,paths);development=objective.SlicedDevelopment(dev,cal);development.cache=BigramCache(development.cache.texts)
 arm=next(a for a in cfg['arms'] if a['id']=='no-match');prior_cache,prior_path=objective.SparseCache,objective.Path
 # Monkeypatch is local to this process; shared module files stay unchanged.
 objective.SparseCache=BigramCache
 objective.Path=lambda path:Path('runs/features-bigram-no-match') if path=='runs' else Path(path)
 report={'configuration':cfg,'scope':'bounded feature/objective interaction; expanded development selection only; original pilot transfer not opened','runs':[]}
 try:
  for seed in [17,29,43]:
   result=objective.train(cfg,arm,seed,train,development);result.update(export(result['checkpoint'],development));report['runs'].append(result)
   Path('eval/feature-experiments-bigram-no-match.json').write_text(json.dumps(report,indent=2)+'\n')
 finally:objective.SparseCache,objective.Path=prior_cache,prior_path
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in cfg['sourceHashes'].items())
 feasible=[r for r in report['runs'] if r['quantizedMetrics']['devNoMatchSemanticRate']<=.05]
 if feasible:
  best=max(feasible,key=lambda r:(r['quantizedMetrics']['devSemanticNdcg5'],r['quantizedMetrics']['devRelevantCoverage']))
  report['selectedArtifact']=best['artifact'];report['selection']='expanded int8 dev semantic nDCG then coverage under dev no-match<=5%, threshold calibration only; transfer unopened'
 Path('eval/feature-experiments-bigram-no-match.json').write_text(json.dumps(report,indent=2)+'\n')
 print('INTERACTION COMPLETE',report.get('selectedArtifact'),flush=True)
if __name__=='__main__':main()
