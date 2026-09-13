"""Portable NumPy specification of the experimental masked objective.

Independent of MLX and used by CPU CI to check behavioral invariants. Only judged
entries must be finite: padding/unjudged entries have no mathematical effect.
"""
import numpy as np


def objective(similarities, positive, judged, *, temperature=.15,
              no_match_weight=0., no_match_target=.35,
              hard_margin_weight=0., hard_margin=.15):
    s=np.asarray(similarities,dtype=np.float64)
    positive=np.asarray(positive,dtype=bool);judged=np.asarray(judged,dtype=bool)
    if s.ndim!=2 or s.shape!=positive.shape or s.shape!=judged.shape:
        raise ValueError('expected equally shaped query/menu matrices')
    if not np.isfinite(temperature) or temperature<=0:
        raise ValueError('temperature must be finite and positive')
    if any(not np.isfinite(x) for x in [no_match_weight,no_match_target,hard_margin_weight,hard_margin]):
        raise ValueError('objective constants must be finite')
    if min(no_match_weight,hard_margin_weight)<0:
        raise ValueError('loss weights cannot be negative')
    if np.any(positive & ~judged):
        raise ValueError('a positive must be judged')
    if np.any(~np.any(judged,axis=1)):
        raise ValueError('each query must have a judged candidate')
    if not np.isfinite(s[judged]).all():
        raise ValueError('judged scores must be finite')
    def logsumexp(values):
        maximum=np.max(values)
        return float(maximum+np.log(np.exp(values-maximum).sum()))
    contrast=[];hard=[];none=[]
    for row,pos,eligible in zip(s,positive,judged):
        if pos.any():
            contrast.append(logsumexp(row[eligible]/temperature)-logsumexp(row[pos]/temperature))
            negatives=eligible & ~pos
            hard.append(max(0.,hard_margin-row[pos].mean()+row[negatives].max()) if negatives.any() else 0.)
        else:
            none.append(max(0.,row[eligible].max()-no_match_target)**2)
    mean=lambda xs:float(np.mean(xs)) if xs else 0.
    return mean(contrast)+hard_margin_weight*mean(hard)+no_match_weight*mean(none)
