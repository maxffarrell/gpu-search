"""Explicit Metal check; deliberately separate from portable unittest discovery."""
import json
from pathlib import Path
import mlx.core as mx
import numpy as np
from training.experiment_training import mlx_objective
from training.objective_reference import objective


def main():
    cases=[([[0.,0.]],[[1,0]],[[1,1]]),([[0.,0.,0.]],[[1,1,0]],[[1,1,1]]),([[.2,.1,1e6]],[[1,0,0]],[[1,1,0]]),([[.6,.1]],[[0,0]],[[1,1]]),([[.2,.1],[.6,.1]],[[1,0],[0,0]],[[1,1],[1,1]])]
    arms=[{'temperature':.15},{'temperature':.07,'hard_margin_weight':.25,'hard_margin':.15},{'temperature':.15,'no_match_weight':.25,'no_match_target':.35}]
    errors=[];masked=[]
    for arm in arms:
        for sims,pos,judged in cases:
            x=mx.array(sims);p=mx.array(pos,dtype=mx.bool_);j=mx.array(judged,dtype=mx.bool_)
            value,gradient=mx.value_and_grad(lambda s:mlx_objective(s,p,j,arm))(x)
            mx.eval(value,gradient)
            error=abs(float(value)-objective(sims,pos,judged,**arm));errors.append(error)
            g=np.array(gradient)
            if not np.isfinite(g).all():raise AssertionError('nonfinite gradients')
            mask=~np.asarray(judged,dtype=bool)
            if mask.any():masked.extend(np.abs(g[mask]).tolist())
            if error>=1e-5:raise AssertionError(error)
    if max(masked,default=0)!=0:raise AssertionError('unjudged entries received gradients')
    result={'framework':'MLX0.31.1 Metal versus portable NumPy reference','cases':len(errors),'maxObjectiveAbsoluteError':max(errors),'maxUnjudgedGradient':max(masked,default=0),'finiteGradients':True,'scope':'single/multiple positives, unjudged adversarial padding, no-match-only and mixed batches across contrastive/hard-margin/no-match arms'}
    Path('eval/training-experiments-objective-parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
