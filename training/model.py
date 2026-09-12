import numpy as np
from training.features import features

def feature_matrices(texts, family='both'):
    w=np.zeros((len(texts),1024),np.float32); c=np.zeros_like(w)
    for i,text in enumerate(texts):
        f=features(text)
        for key,out in [('wordIds',w),('charIds',c)]:
            ids=f[key]
            if ids:
                np.add.at(out[i],ids,1/len(ids))
    if family=='word': c[:]=0
    if family=='char': w[:]=0
    return w,c

def encode_numpy(params, texts, family='both', stages=False):
    w,c=feature_matrices(texts,family)
    x=np.float32(.5)*(w@params['word']+c@params['char'])
    h=np.tanh(x@params['W1'].T+params['b1']) if 'W1' in params else x
    z=h@params['W2'].T+params['b2'] if 'W2' in params else h
    norms=np.linalg.norm(z,axis=-1,keepdims=True)
    v=z/np.maximum(norms,1e-8)
    return {'pooled':x,'hidden':h,'projected':z,'norms':norms,'vectors':v} if stages else v

def candidate_vectors(params,candidates,family='both'):
    result=[]
    for item in candidates:
        vecs=encode_numpy(params,[item['label'],*item.get('aliases',[])],family)
        valid=vecs[np.linalg.norm(vecs,axis=1)>=1e-8]
        v=valid.mean(axis=0) if len(valid) else np.zeros(params['word'].shape[1],np.float32)
        if item.get('context'): v=v+.25*encode_numpy(params,[item['context']],family)[0]
        norm=np.linalg.norm(v); result.append(v/max(norm,1e-8))
    return np.array(result)
