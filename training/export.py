"""Versioned experimental exporter; does not approve a model for production."""
import argparse,hashlib,json,math,struct
from pathlib import Path
import numpy as np
from training.features import VERSION,features
from training.model import encode_numpy

def quantize(values,bits):
    if bits not in (6,8):raise ValueError('bits must be 6 or 8')
    values=np.asarray(values,dtype=np.float32)
    if not np.isfinite(values).all():raise ValueError('nonfinite weights')
    Q=2**(bits-1)-1;maximum=float(np.max(np.abs(values))) if values.size else 0
    scale=np.float32(maximum/Q if maximum else 1)
    q=np.clip(np.sign(values)*np.floor(np.abs(values/scale)+.5),-Q,Q).astype(np.int8)
    return q,scale

def pack(codes,bits):
    codes=np.asarray(codes).ravel()
    if bits==8:
        if np.any(codes<-127) or np.any(codes>127):raise ValueError('invalid int8 code')
        return codes.astype(np.int8).tobytes()
    if bits!=6 or np.any(codes<-31) or np.any(codes>31):raise ValueError('invalid int6 code')
    result=bytearray(math.ceil(len(codes)*6/8))
    for i,code in enumerate(codes):
        value=int(code)+31;bit=i*6
        for j in range(6):result[(bit+j)//8]|=((value>>j)&1)<<((bit+j)%8)
    return bytes(result)

def unpack(payload,count,bits):
    if len(payload)!=math.ceil(count*bits/8):raise ValueError('payload length mismatch')
    if bits==8:
        a=np.frombuffer(payload,dtype=np.int8).copy()
        if np.any(a==-128):raise ValueError('reserved int8 code')
        return a
    if bits!=6:raise ValueError('unsupported bits')
    out=[]
    for i in range(count):
        code=sum(((payload[(i*6+j)//8]>>((i*6+j)%8))&1)<<j for j in range(6))
        if code==63:raise ValueError('reserved int6 code')
        out.append(code-31)
    if count*6%8 and payload[-1]>>(count*6%8):raise ValueError('nonzero trailing bits')
    return np.array(out,dtype=np.int8)

def export(checkpoint,bits,output):
    cp=Path(checkpoint);meta=json.loads((cp/'training.json').read_text());params=dict(np.load(cp/'weights.npz'));payload=bytearray();tensors=[];quantized={}
    for name,values in params.items():
        codes,scale=quantize(values,bits);blob=pack(codes,bits)
        tensors.append({'name':name,'shape':list(values.shape),'elementCount':values.size,'bits':bits,'scale':float(scale),'scaleF32LE':struct.pack('<f',scale).hex(),'byteOffset':len(payload),'byteLength':len(blob)})
        payload.extend(blob);quantized[name]=(unpack(blob,values.size,bits).reshape(values.shape)*scale).astype(np.float32)
    manifest={'formatVersion':'gpu-search-experimental-v1','featureVersion':VERSION,'normalizationVersion':'nfkc-ascii-v1','candidateCompositionVersion':'mean-alias-context025-v1','architecture':meta['architecture'],'dimension':meta['dimension'],'featureFamily':meta['family'],'modelId':cp.name,'byteOrder':'little-endian','tensors':tensors,'payloadSha256':hashlib.sha256(payload).hexdigest(),'payloadBytes':len(payload),'validatedSemanticCutoff':None,'status':'EXPERIMENTAL ONLY; release quality gate not passed; do not load in default search'}
    out=Path(output);out.mkdir(parents=True,exist_ok=True);(out/'weights.bin').write_bytes(payload);(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    texts=['Profile','coworkers','C++','myProfile',''];stages=encode_numpy(quantized,texts,meta['family'],stages=True)
    (out/'numerical-fixtures.json').write_text(json.dumps([{'text':t,'features':features(t),**{k:v[i].tolist() for k,v in stages.items()}} for i,t in enumerate(texts)],indent=2)+'\n')
    return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--bits',type=int,choices=[6,8],default=8);p.add_argument('--output',default='packages/model/experimental');p.add_argument('--experimental',action='store_true');a=p.parse_args()
    if not a.experimental:p.error('No production model passed the gate. Add --experimental to export a research checkpoint explicitly.')
    print(json.dumps(export(a.checkpoint,a.bits,a.output),indent=2))
if __name__=='__main__':main()
