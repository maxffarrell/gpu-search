"""Bounded frozen-candidate evaluation; sealed input is opt-in and access is receipted.

Default execution NEVER reads data/test.jsonl. --sealed additionally requires a
prewritten selection record matching both candidate manifest and payload hashes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
from training.evaluate_expanded import Evaluator, read_rows, load_artifact, calibrate, rankings, summarize, bootstrap


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def asset_metadata(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    payload_hash=digest(directory/'weights.bin')
    if payload_hash!=manifest['payloadSha256']:raise ValueError('Candidate payload differs from manifest hash')
    return {'directory':str(directory),'modelId':manifest['modelId'],'manifestSha256':digest(directory/'manifest.json'),'payloadSha256':payload_hash,'payloadBytes':(directory/'weights.bin').stat().st_size,'featureVersion':manifest['featureVersion'],'dimension':manifest['dimension'],'quantizationBits':sorted({t['bits'] for t in manifest['tensors']})}


def portable_scores(module,params,rows):
    texts=sorted({text for row in rows for text in [row['query'],*[text for c in row['candidates'] for text in [c['label'],*c.get('aliases',[]),c.get('context','')]]]})
    encoded=module.encode_numpy(params,texts);lookup=dict(zip(texts,encoded));cache={};output=[]
    for row in rows:
        query=lookup[row['query']];scores=[]
        for c in row['candidates']:
            key=(c['label'],tuple(c.get('aliases',[])),c.get('context',''))
            if key not in cache:
                valid=[lookup[t] for t in [c['label'],*c.get('aliases',[])] if np.linalg.norm(lookup[t])>=1e-8]
                vector=np.mean(valid,axis=0) if valid else np.zeros_like(query)
                if c.get('context'):vector=vector+np.float32(.25)*lookup[c['context']]
                norm=np.linalg.norm(vector);cache[key]=vector/max(norm,1e-8)
            vector=cache[key]
            scores.append(float(query@vector) if np.linalg.norm(query)>=1e-8 and np.linalg.norm(vector)>=1e-8 else -np.inf)
        output.append(scores)
    return output


def model_scores(directory,rows,adapter=None):
    params,manifest=load_artifact(directory)
    if adapter:
        module=importlib.import_module(adapter)
        values=module.score_rows(params,rows,manifest) if hasattr(module,'score_rows') else portable_scores(module,params,rows)
    else:
        if manifest['featureVersion']=='gpu-search-features-bigram25-v1':
            if manifest['architecture']!='pooled' or manifest['featureFamily']!='both':raise ValueError('Unsupported bigram architecture/family')
            values=portable_scores(importlib.import_module('training.features_bigram'),params,rows)
        elif manifest['featureVersion']=='gpu-search-features-v1':
            values=Evaluator([],rows).scores(params,manifest['featureFamily'])
        else:
            raise ValueError('Nonstandard feature version requires explicit --adapter module')
    if len(values)!=len(rows) or any(len(v)!=len(r['candidates']) for r,v in zip(rows,values)):
        raise ValueError('Scoring adapter returned mismatched row/candidate counts')
    if any(np.isnan(v).any() or np.isposinf(v).any() for v in map(np.asarray,values)):
        raise ValueError('Scoring adapter returned invalid scores; use -inf only for zero embeddings')
    return values


def add_no_match_counts(metric):
    count=metric['noMatchCount']
    for key in ['noMatchAnyRate','noMatchSemanticRate']:
        metric[key.replace('Rate','ReturnedCount')]=round(metric[key]*count) if metric[key] is not None else 0
    return metric


def describe(evaluator,indices,raw=False):
    chosen=evaluator.semantic
    result={'overall':add_no_match_counts(summarize(evaluator.rows,indices,None if raw else evaluator.tiers)),
            'semanticOnly':add_no_match_counts(summarize([evaluator.rows[i] for i in chosen],[indices[i] for i in chosen],None if raw else [evaluator.tiers[i] for i in chosen])),'slices':{}}
    for field in ['evaluation_slice','query_kind']:
        for name in sorted({r.get(field,'unspecified') for r in evaluator.rows}):
            take=[i for i,r in enumerate(evaluator.rows) if r.get(field,'unspecified')==name]
            result['slices'][f'{field}:{name}']=add_no_match_counts(summarize([evaluator.rows[i] for i in take],[indices[i] for i in take],None if raw else [evaluator.tiers[i] for i in take]))
    return result


def compare_partition(rows,candidate_scores,baseline_scores,candidate_cutoff,baseline_cutoff):
    evaluator=Evaluator([],rows);systems={};ranked={}
    for name,scores,cutoff in [('candidate',candidate_scores,candidate_cutoff),('baseline',baseline_scores,baseline_cutoff)]:
        for mode in ['calibrated','raw','combined-before-cutoff']:
            key=f'{name}-{mode}'
            indices=rankings(rows,scores,cutoff if mode=='calibrated' else -np.inf,evaluator.tiers,raw=mode=='raw')
            ranked[key]=indices;systems[key]=describe(evaluator,indices,raw=mode=='raw')
    comparisons={mode:bootstrap(rows,ranked[f'candidate-{mode}'],ranked[f'baseline-{mode}'],evaluator.semantic) for mode in ['calibrated','raw','combined-before-cutoff']}
    positive=[i for i,r in enumerate(rows) if any(g>0 for g in r['relevance'].values())]
    comparisons['overall-calibrated']=bootstrap(rows,ranked['candidate-calibrated'],ranked['baseline-calibrated'],positive)
    return {'count':len(rows),'semanticCount':len(evaluator.semantic),'systems':systems,'pairedCandidateMinusBaseline':comparisons,'perQuery':[{'id':r['id'],'top5':{name:indices[i][:5] for name,indices in ranked.items()}} for i,r in enumerate(rows)]}


def claim_sealed_access(selection_record,candidate_metadata,receipt_path='eval/sealed-test-access.json'):
    if not selection_record:raise ValueError('--sealed requires --selection-record')
    selection=json.loads(Path(selection_record).read_text())
    if selection.get('selectionFrozen') is not True:raise ValueError('Selection must be frozen before sealed access')
    for key in ['payloadSha256','manifestSha256']:
        if selection.get(key)!=candidate_metadata[key]:raise ValueError(f'Frozen selection {key} mismatch')
    receipt={'accessedAt':datetime.now(timezone.utc).isoformat(),'selectionRecord':str(selection_record),'selectionRecordSha256':digest(selection_record),'candidate':candidate_metadata,'dataset':'data/test.jsonl','status':'Access claimed before reading; no automatic reruns','scope':'Existing synthetic sealed pilot, not human-reviewed final evidence'}
    # Exclusive creation prevents alternate output prefixes from reopening the same test silently.
    with Path(receipt_path).open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    return receipt


def evaluate(candidate='packages/model/candidate',baseline='packages/model/candidate',adapter=None,baseline_adapter=None,sealed=False,selection_record=None):
    candidate_meta=asset_metadata(candidate);baseline_meta=asset_metadata(baseline)
    calibration_path='data/expanded/calibration.jsonl';calibration=read_rows(calibration_path)
    # Validate adapters and artifacts, and fix thresholds, before requesting any sealed access.
    ccut=calibrate(calibration,model_scores(candidate,calibration,adapter))
    bcut=calibrate(calibration,model_scores(baseline,calibration,baseline_adapter))
    partitions={'expanded-dev':read_rows('data/expanded/dev.jsonl'),'original-dev-transfer':read_rows('data/dev.jsonl')}
    paths=[calibration_path,'data/expanded/dev.jsonl','data/dev.jsonl']
    receipt=None
    if sealed:
        receipt=claim_sealed_access(selection_record,candidate_meta)
        partitions['sealed-synthetic-test']=read_rows('data/test.jsonl');paths.append('data/test.jsonl')
    output={}
    for name,rows in partitions.items():
        output[name]=compare_partition(rows,model_scores(candidate,rows,adapter),model_scores(baseline,rows,baseline_adapter),ccut,bcut)
    floors_path='eval/expanded-quality-floors.json';floors=json.loads(Path(floors_path).read_text())
    expanded=output['expanded-dev']['systems']['candidate-calibrated'];transfer=output['original-dev-transfer']['systems']
    checks={
        'semanticNdcgFloor':expanded['semanticOnly']['ndcg5']>=floors['minimumSemanticNdcg5'],
        'semanticCoverageFloor':expanded['semanticOnly']['coverage']>=floors['minimumSemanticCoverage'],
        'noMatchBudget':expanded['overall']['noMatchSemanticRate']<=floors['maximumNoMatchSemanticRate'],
        'transferRawOverallFloor':transfer['candidate-raw']['overall']['ndcg5']>=floors['minimumTransferRawOverallNdcg5'],
        'transferCombinedOverallFloor':transfer['candidate-calibrated']['overall']['ndcg5']>=floors['minimumTransferCombinedOverallNdcg5'],
        'expandedOverallRegression':expanded['overall']['ndcg5']>=output['expanded-dev']['systems']['baseline-calibrated']['overall']['ndcg5']-.01,
        'transferOverallRegression':transfer['candidate-calibrated']['overall']['ndcg5']>=transfer['baseline-calibrated']['overall']['ndcg5']-.01,
    }
    code_paths=['training/evaluate_release_candidate.py','training/evaluate_expanded.py','training/features.py','training/model.py']
    if any(m['featureVersion']=='gpu-search-features-bigram25-v1' for m in [candidate_meta,baseline_meta]):code_paths.append('training/features_bigram.py')
    return {'scope':'Frozen candidate comparison on weak-label development and synthetic transfer; no reviewed final quality claim','candidate':candidate_meta,'baseline':baseline_meta,'adapters':{'candidate':adapter,'baseline':baseline_adapter},'cutoffs':{'candidate':ccut,'baseline':bcut,'fitPartition':calibration_path},'partitions':output,'floorChecks':checks,'allRegressionFloorsPassed':all(checks.values()),'frozenFloors':floors,'sourceSha256':{path:digest(path) for path in paths+[floors_path]+code_paths},'sealedAccess':receipt,'sealedExecuted':sealed,'reviewedFinalGatePassed':False,'uncertainty':'Paired query bootstrap is descriptive; ignores intent-family clustering and prior development selection'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',default='packages/model/candidate');parser.add_argument('--baseline',default='packages/model/candidate');parser.add_argument('--adapter');parser.add_argument('--baseline-adapter');parser.add_argument('--output',default='eval/release-candidate-report.json');parser.add_argument('--sealed',action='store_true');parser.add_argument('--selection-record');args=parser.parse_args()
    report=evaluate(args.candidate,args.baseline,args.adapter,args.baseline_adapter,args.sealed,args.selection_record)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'candidate':report['candidate'],'checks':report['floorChecks'],'sealedExecuted':report['sealedExecuted'],'output':args.output},indent=2))


if __name__=='__main__':main()
