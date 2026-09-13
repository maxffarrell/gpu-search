"""Portable NumPy reference for the isolated ordered-bigram pooled16 experiment.

This is a new feature contract, not a change to gpu-search-features-v1.
"""
import numpy as np
from training.features import features as unigram_features, tokens, hash_bytes
VERSION='gpu-search-features-bigram25-v1'

def features(text):
    base=unigram_features(text)
    ts=[token[:64] for token in tokens(text)[:32]]
    bigrams=[]
    for left,right in zip(ts,ts[1:]):
        a,b=left.encode('utf-8'),right.encode('utf-8')
        blob=b'b:'+len(a).to_bytes(4,'little')+a+len(b).to_bytes(4,'little')+b
        bigrams.append(hash_bytes(blob))
    return {**base,'bigramIds':bigrams,'bigramCount':len(bigrams)}

def encode_numpy(params,texts,stages=False):
    pooled=[]
    for text in texts:
        f=features(text)
        def mean(name,ids):
            return np.mean(params[name][ids],axis=0,dtype=np.float32) if ids else np.zeros(params['word'].shape[1],np.float32)
        word=mean('word',f['wordIds'])
        if f['bigramIds']:
            word=np.float32(.75)*word+np.float32(.25)*mean('word',f['bigramIds'])
        pooled.append(np.float32(.5)*(word+mean('char',f['charIds'])))
    x=np.asarray(pooled,dtype=np.float32).reshape(-1,params['word'].shape[1])
    norms=np.linalg.norm(x,axis=-1,keepdims=True)
    vectors=x/np.maximum(norms,np.float32(1e-8))
    return {'pooled':x,'norms':norms,'vectors':vectors} if stages else vectors

def candidate_vectors(params,candidates):
    result=[]
    for candidate in candidates:
        vectors=encode_numpy(params,[candidate['label'],*candidate.get('aliases',[])])
        valid=vectors[np.linalg.norm(vectors,axis=1)>=1e-8]
        vector=valid.mean(axis=0) if len(valid) else np.zeros(params['word'].shape[1],np.float32)
        if candidate.get('context'):
            vector=vector+np.float32(.25)*encode_numpy(params,[candidate['context']])[0]
        norm=np.linalg.norm(vector)
        result.append(vector/max(norm,1e-8))
    return np.asarray(result,dtype=np.float32)
