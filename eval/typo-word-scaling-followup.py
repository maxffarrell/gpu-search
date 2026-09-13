"""Bounded second-stage follow-up after the initial five fixed scales; development partitions only, no export or fitting."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from training.evaluate_expanded import Evaluator, load_artifact, read_rows, calibrate, rankings, summarize
from training.evaluate_navigation import metrics

ARTIFACT = Path('packages/model/experiments/navigation-align0p5-seed29')
PATHS = {'calibration': 'data/expanded/calibration.jsonl', 'expanded': 'data/expanded/dev.jsonl', 'typo': 'data/typo/dev.jsonl', 'navigation': 'data/navigation-splits/dev.jsonl'}
RATIOS = [1, .8, .85, .9]
OUTPUT = Path('eval/typo-word-scaling-followup.json')
GATES = {'expandedSemanticNdcg5': .45, 'expandedCoverage': .48, 'expandedNoMatchMaximum': .05, 'expandedOverallNdcg5': .447708, 'navigationSemanticRawNdcg5': .5113}

def main():
    params, manifest = load_artifact(ARTIFACT)
    rows = {name: read_rows(path) for name,path in PATHS.items()}
    expanded = Evaluator(rows['calibration'], rows['expanded'])
    typo = Evaluator([], rows['typo'])
    navigation = Evaluator([], rows['navigation'])
    results=[]
    for ratio in RATIOS:
        scaled={**params, 'word':(params['word'] * np.float32(ratio)).astype(np.float32)}
        expanded_result=expanded.evaluate(scaled)
        ts=typo.scores(scaled)
        # Raw complete cosine order; no lexical or minimum-query-length filtering.
        tr=[sorted(range(len(s)), key=lambda j:(-s[j],j)) for s in ts]
        ns=navigation.scores(scaled)
        nr=[sorted(range(len(s)), key=lambda j:(-s[j],j)) for s in ns]
        typo_result=metrics(rows['typo'],tr,list(range(len(tr))))
        short_indices=[i for i,row in enumerate(rows['typo']) if len(row['cleanLabel'].split())<=2]
        short_result=metrics(rows['typo'],tr,short_indices)
        nav_result=metrics(rows['navigation'],nr,navigation.semantic)
        passes={
            'expandedSemanticNdcg5':expanded_result['semanticNdcg5']>=GATES['expandedSemanticNdcg5'],
            'expandedCoverage':expanded_result['semanticCoverage']>=GATES['expandedCoverage'],
            'expandedNoMatchMaximum':expanded_result['noMatchSemanticRate']<=GATES['expandedNoMatchMaximum'],
            'expandedOverallNdcg5':expanded_result['overallNdcg5']>=GATES['expandedOverallNdcg5'],
            'navigationSemanticRawNdcg5':nav_result['ndcg5']>=GATES['navigationSemanticRawNdcg5'],
        }
        result={'wordScale':ratio,'typoRaw':typo_result,'typoShortRaw':short_result,'selectionScore':(typo_result['success1']+short_result['success1'])/2,'expanded':expanded_result,'navigationSemanticRaw':nav_result,'gates':passes,'allGatesPassed':all(passes.values())}
        results.append(result)
        print(json.dumps({'ratio':ratio,'typoTop1':typo_result['success1'],'typoTop5':typo_result['success5'],'expandedNdcg':expanded_result['semanticNdcg5'],'expandedFP':expanded_result['noMatchSemanticRate'],'navigationNdcg':nav_result['ndcg5'],'passed':all(passes.values())}),flush=True)
    report={'generatedAt':datetime.now(timezone.utc).isoformat(),'modelId':manifest['modelId'],'modelSha256':manifest['payloadSha256'],'sourceHashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in PATHS.values()},'readPartitions':PATHS,'frozenRatios':RATIOS,'stage':'Follow-up authorized after .75 missed coverage; baseline1 plus .8/.85/.9. Selection maximizes mean(short-label top1, all top1) among all gate-passing arms.','gates':GATES,'rawWeightBytes':manifest['payloadBytes'],'method':'Dequantized int8 word table multiplied globally by fixed ratio; char table unchanged. Existing exact pooled normalization and alias/context composition. Each ratio has its own expanded-calibration-only cutoff. No training or held-out/custom-query/diagnostic access. Ratios can be represented by tensor scale metadata with identical 32768-byte int8 payload; zero ratio requires zero word codes and positive scale. No artifacts exported.','limitations':['Development-selected ablation, not independent generalization evidence.','Typo labels are synthetic one-edit origins; other labels are not reviewed semantic negatives.','XFCE only documented positives; no no-match metric is available.','No change to lexical ranking, runtime or shipped assets.'],'results':results}
    OUTPUT.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
