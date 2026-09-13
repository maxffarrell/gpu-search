"""Tiny rejection-rule ablations on fixed weights; no test set or runtime changes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from training.evaluate_expanded import Evaluator, read_rows, load_artifact, summarize, bootstrap, rankings, calibrate
from training.features import normalize

SETTINGS={
    'version':'candidate-linear-rejector-v1',
    'features':{'score':['score'],'best-margin':['score','bestMargin'],'mean-margin':['score','meanMargin'],'both-margins':['score','bestMargin','meanMargin'],'mean-dispersion':['score','meanMargin','dispersion']},
    'ridge':[.1,1.,10.], 'calibrationFalsePositiveBudgets':[.01,.025,.05],
    'maximumDevNoMatchRate':.05,'minimumDevOverallDelta':-.01,'newtonSteps':50,
    'fit':'all calibration candidate binary relevance>=2 labels; class-balanced regularized logistic objective',
    'threshold':'calibration no-match menu maximum linear statistics only',
    'selection':'highest dev semantic nDCG among rules meeting5% dev no-match and baseline overall regression allowance; prefer fewer coefficients on ties',
    'menuSize':'all observed expanded menus have10 candidates; coefficient is not identifiable and not fitted',
    'multiPositive':'candidate-wise correctness target; best-margin optional, never mandatory positive gap; no top1-only restriction',
    'noReviewedFinalTest':True,
}


def feature_rows(scores):
    """Per-candidate features preserve permutation and permit tied relevant candidates."""
    values=np.asarray(scores,np.float64);finite=values[np.isfinite(values)]
    result=[]
    for i,score in enumerate(values):
        other=np.delete(values,i);other=other[np.isfinite(other)]
        if not np.isfinite(score):result.append([0.,0.,0.,0.]);continue
        result.append([score,score-max(other) if len(other) else 0.,score-float(np.mean(other)) if len(other) else 0.,float(np.std(finite)) if len(finite) else 0.])
    return np.asarray(result,np.float64)


def fit_linear(x,y,ridge,steps=50):
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-8]=1.
    design=np.column_stack([np.ones(len(x)),(x-mean)/scale]);coef=np.zeros(design.shape[1])
    positive=max(1,int(sum(y)));negative=max(1,len(y)-positive)
    weights=np.where(y, len(y)/(2*positive),len(y)/(2*negative))
    penalty=np.diag([0.,*([ridge]*(design.shape[1]-1))])
    for _ in range(steps):
        p=1/(1+np.exp(-np.clip(design@coef,-35,35)))
        gradient=design.T@(weights*(p-y))+penalty@coef
        curvature=(design.T*(weights*p*(1-p)))@design+penalty+np.eye(design.shape[1])*1e-9
        step=np.linalg.solve(curvature,gradient);coef-=step
        if np.max(np.abs(step))<1e-8:break
    raw=coef[1:]/scale
    return {'bias':float(coef[0]-raw@mean),'weights':[float(value) for value in raw]}


def statistics(scores,rule):
    order=['score','bestMargin','meanMargin','dispersion'];columns=[order.index(k) for k in rule['features']]
    result=[]
    for menu in scores:
        value=feature_rows(menu)[:,columns]@np.asarray(rule['weights'])+rule['bias']
        value[~np.isfinite(menu)]=-np.inf;result.append(value.tolist())
    return result


def threshold(rows,values,budget):
    maxima=sorted([max(v,default=-np.inf) for r,v in zip(rows,values) if not any(r['relevance'].values())],reverse=True)
    if not maxima:raise ValueError('Calibration requires no-match menus')
    value=maxima[min(len(maxima)-1,math.floor(budget*len(maxima)))]
    if not np.isfinite(value):return float(np.finfo(np.float64).max)
    return float(np.nextafter(value,np.inf))


def apply_rule(rows,scores,tiers,rule):
    evidence=statistics(scores,rule);result=[]
    for row,cosines,decision,lexical in zip(rows,scores,evidence,tiers):
        pairs=[]
        for i,(cosine,value,tier) in enumerate(zip(cosines,decision,lexical)):
            if tier[0]:pairs.append((tier,i))
            elif len(normalize(row['query']))>=3 and np.isfinite(cosine) and value>=rule['threshold']:
                # Learned rejection changes eligibility, never semantic ordering or hard tiers.
                pairs.append(((1,0,float(cosine)),i))
        result.append([i for _,i in sorted(pairs,key=lambda p:(tuple(-v for v in p[0]),p[1]))])
    return result


def metrics(evaluator,indices):
    selected=evaluator.semantic
    result={'overall':summarize(evaluator.rows,indices,evaluator.tiers),
            'semanticOnly':summarize([evaluator.rows[i] for i in selected],[indices[i] for i in selected],[evaluator.tiers[i] for i in selected]),'slices':{}}
    for name in sorted({r.get('evaluation_slice','original-pilot-transfer') for r in evaluator.rows}):
        part=[i for i,r in enumerate(evaluator.rows) if r.get('evaluation_slice','original-pilot-transfer')==name]
        result['slices'][name]=summarize([evaluator.rows[i] for i in part],[indices[i] for i in part],[evaluator.tiers[i] for i in part])
    return result


def experiment(model_directory,prefix='eval/calibration',skip_transfer=False):
    Path(prefix+'-settings.json').write_text(json.dumps(SETTINGS,indent=2)+'\n')
    calibration=read_rows('data/expanded/calibration.jsonl');dev=read_rows('data/expanded/dev.jsonl')
    model,manifest=load_artifact(model_directory)
    evaluator=Evaluator(calibration,dev);all_scores=evaluator.scores(model,manifest['featureFamily']);cal_scores=all_scores[:len(calibration)];dev_scores=all_scores[len(calibration):]
    original_cutoff=calibrate(calibration,cal_scores)
    baseline=rankings(dev,dev_scores,original_cutoff,evaluator.tiers);baseline_metrics=metrics(evaluator,baseline)
    column_order=['score','bestMargin','meanMargin','dispersion']
    x=np.concatenate([feature_rows(scores) for scores in cal_scores]);y=np.asarray([r['relevance'].get(c['id'],0)>=2 for r in calibration for c in r['candidates']],np.float64)
    finite=np.concatenate([np.isfinite(scores) for scores in cal_scores]);x=x[finite];y=y[finite]
    experiments=[];best=None;best_indices=None
    for name,names in SETTINGS['features'].items():
        columns=[column_order.index(k) for k in names]
        for ridge in SETTINGS['ridge']:
            fitted=fit_linear(x[:,columns],y,ridge,SETTINGS['newtonSteps'])
            rule={'features':names,**fitted}
            cal_evidence=statistics(cal_scores,rule)
            for budget in SETTINGS['calibrationFalsePositiveBudgets']:
                current={**rule,'threshold':threshold(calibration,cal_evidence,budget)}
                ranked=apply_rule(dev,dev_scores,evaluator.tiers,current);observed=metrics(evaluator,ranked)
                admissible=observed['overall']['noMatchSemanticRate']<=SETTINGS['maximumDevNoMatchRate'] and observed['overall']['ndcg5']>=baseline_metrics['overall']['ndcg5']+SETTINGS['minimumDevOverallDelta']
                entry={'name':name,'ridge':ridge,'calibrationBudget':budget,'rule':current,'metrics':observed,'admissible':admissible}
                experiments.append(entry)
                if admissible and (best is None or (observed['semanticOnly']['ndcg5'],-len(names))>(best['metrics']['semanticOnly']['ndcg5'],-len(best['rule']['features']))):best=entry;best_indices=ranked
    report={'scope':'calibration-fitted rejection on existing int8 weights, weak-label development selection only','modelId':manifest['modelId'],'modelSha256':manifest['payloadSha256'],'baselineCutoff':original_cutoff,'baseline':baseline_metrics,'experiments':experiments,'selected':best,'candidateCountObserved':sorted({len(r['candidates']) for r in calibration+dev}),
            'limitations':['Fitting and threshold calibration reuse calibration labels; dev selects45 finite settings','Dataset relevance is source-derived and unreviewed; no independent final claim','10-candidate menus do not identify size dependence; other menu sizes unsupported','Most supervision has one positive per menu; candidate rule permits multiple positives but real multi-positive quality is unmeasured','Query bootstrap is descriptive after dev selection and ignores source-family dependence']}
    if best:
        report['pairedVersusBaseline']=bootstrap(dev,best_indices,baseline,evaluator.semantic)
        if not skip_transfer:
            transfer=Evaluator([],read_rows('data/dev.jsonl'));transfer_scores=transfer.scores(model,manifest['featureFamily'])
            transfer_baseline=rankings(transfer.rows,transfer_scores,original_cutoff,transfer.tiers)
            transfer_new=apply_rule(transfer.rows,transfer_scores,transfer.tiers,best['rule'])
            report['postSelectionTransfer']={'baseline':metrics(transfer,transfer_baseline),'selected':metrics(transfer,transfer_new),'paired':bootstrap(transfer.rows,transfer_new,transfer_baseline,transfer.semantic)}
        else:report['postSelectionTransfer']={'status':'Not read or scored; candidate selection still in progress'}
        artifact={'formatVersion':'candidate-linear-rejector-v1','modelSha256':manifest['payloadSha256'],'candidateCountSupported':10,'fallbackCosineCutoff':original_cutoff,**best['rule'],'status':'experimental weak-label dev selection; not confidence probabilities'}
        Path(prefix+'-candidate.json').write_text(json.dumps(artifact,indent=2)+'\n')
    report['datasetSha256']={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in ['data/expanded/calibration.jsonl','data/expanded/dev.jsonl']+([] if skip_transfer else ['data/dev.jsonl'])}
    Path(prefix+'-report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',default='packages/model/candidate');parser.add_argument('--prefix',default='eval/calibration');parser.add_argument('--skip-transfer',action='store_true');args=parser.parse_args()
    result=experiment(args.model,args.prefix,args.skip_transfer)
    print(json.dumps({k:v for k,v in result.items() if k not in ['experiments']},indent=2))


if __name__=='__main__':main()
