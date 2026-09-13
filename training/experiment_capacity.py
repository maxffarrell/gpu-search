"""24-dimensional capacity pilot with fixed-cutoff quantization comparison.

Uses the same no-match objective as the completed 16-dimensional objective arm.
Does not tune a cutoff on development or evaluate original transfer data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from training.experiment_training import train,SlicedDevelopment
from training.train_expanded import read,validate_rows
from training.evaluate_expanded import Evaluator,rankings,summarize
from training.export import export,unpack


def quantized_params(folder):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text());blob=(folder/'weights.bin').read_bytes()
    return {t['name']:(unpack(blob[t['byteOffset']:t['byteOffset']+t['byteLength']],t['elementCount'],t['bits']).reshape(t['shape'])*np.float32(t['scale'])).astype(np.float32) for t in manifest['tensors']}


def compare(checkpoint,cal,dev):
    evaluator=Evaluator(cal,dev)
    float_params=dict(np.load(Path(checkpoint)/'weights.npz'))
    ref=evaluator.evaluate(float_params,detail=True)
    cutoff=ref['cutoff']
    output={'checkpoint':checkpoint,'floatCutoff':cutoff,'cutoffPolicy':'Frozen float cutoff fitted on calibration only; shared across float/int8/int6','float':{k:v for k,v in ref.items() if k not in ('scores','rankings')}}
    for bits in [8,6]:
        target=f"packages/model/experiments/{Path(checkpoint).name}-int{bits}"
        manifest=export(checkpoint,bits,target)
        params=quantized_params(target)
        scores=evaluator.scores(params)[len(cal):]
        ranked=rankings(dev,scores,cutoff,evaluator.tiers)
        overall=summarize(dev,ranked,evaluator.tiers)
        sem_ids=evaluator.semantic
        semantic=summarize([dev[i] for i in sem_ids],[ranked[i] for i in sem_ids],[evaluator.tiers[i] for i in sem_ids])
        slices={}
        for label in sorted({r.get('evaluation_slice','unspecified') for r in dev}):
            ids=[i for i in sem_ids if dev[i].get('evaluation_slice','unspecified')==label]
            slices[label]=summarize([dev[i] for i in ids],[ranked[i] for i in ids],[evaluator.tiers[i] for i in ids])
        output[f'int{bits}']={'artifact':target,'payloadBytes':manifest['payloadBytes'],'overall':overall,'semanticOnly':semantic,'slices':slices,'semanticNdcgLoss':ref['semanticNdcg5']-semantic['ndcg5'],'passesQuantizationLossGate':ref['semanticNdcg5']-semantic['ndcg5']<=.01,'passesDevNoMatchGate':overall['noMatchSemanticRate']<=.05}
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/experiments-capacity.yaml');p.add_argument('--tag');args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    if args.tag:
        if not args.tag.replace('-','').replace('_','').isalnum():p.error('invalid tag')
        cfg['tag']=args.tag
    if cfg['dimension']!=24:raise ValueError('Capacity pilot freezes dimension24')
    root=Path(cfg['data_dir']);paths=[root/f'{s}.jsonl' for s in ['train','dev','calibration']]
    cfg['sourceHashes']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    rows,dev,cal=map(read,paths);validate_rows(dev+cal)
    evaluator=SlicedDevelopment(dev,cal)
    out=Path('eval')/('training-experiments-capacity'+('-'+args.tag if args.tag else '')+'.json')
    report={'status':'weak-label dev-selected capacity ablation; no original transfer data evaluated','configuration':cfg,'baselineObjective':'Same no-match loss and feature contract as experiment-no-match seeds17/29/43; those selected epochs all<30','runs':[],'quantization':[]}
    for seed in cfg['seeds']:
        result=train(cfg,cfg['arms'][0],seed,rows,evaluator);report['runs'].append(result)
        report['quantization'].append(compare(result['checkpoint'],cal,dev))
        if any(hashlib.sha256(path.read_bytes()).hexdigest()!=cfg['sourceHashes'][str(path)] for path in paths):raise RuntimeError('Data changed')
        out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
