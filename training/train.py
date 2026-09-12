import argparse, json, time, platform
from pathlib import Path
import numpy as np
import yaml
import mlx.core as mx
import mlx.optimizers as optim
from training.model import feature_matrices, encode_numpy, candidate_vectors

def read(split): return [json.loads(l) for l in Path(f'data/{split}.jsonl').read_text().splitlines()]
def train_one(cfg,architecture,seed,dimension=16,family='both',loss_kind='contrastive'):
    mx.random.seed(seed)
    params={'word':mx.random.normal((1024,dimension))*.1,'char':mx.random.normal((1024,dimension))*.1}
    if architecture=='projected': params.update(W1=mx.random.normal((24,dimension))*.2,b1=mx.zeros(24),W2=mx.random.normal((dimension,24))*.2,b2=mx.zeros(dimension))
    rows=[r for r in read('train') if max(r['relevance'].values())>=2]
    texts=[]; qidx=[]; cidx=[]; positives=[]; eligible=[]
    for r in rows:
        qidx.append(len(texts)); texts.append(r['query']); inds=[]
        for c in r['candidates']:
            if c.get('aliases') or c.get('context'): raise ValueError('This prepared synthetic experiment uses label-only candidates; metadata training requires new prepared batches')
            inds.append(len(texts));texts.append(c['label'])
        cidx.append(inds);positives.append([r['relevance'].get(c['id'],-1)>=2 for c in r['candidates']]);eligible.append([c['id'] in r['relevance'] for c in r['candidates']])
    wf,cf=map(mx.array,feature_matrices(texts,family));qi=mx.array(qidx);ci=mx.array(cidx);pos=mx.array(positives);judged=mx.array(eligible)
    def loss(p):
        x=.5*(wf@p['word']+cf@p['char'])
        if architecture=='projected': x=mx.tanh(x@p['W1'].T+p['b1'])@p['W2'].T+p['b2']
        x=x/mx.maximum(mx.sqrt(mx.sum(x*x,axis=-1,keepdims=True)),1e-8)
        sim=mx.sum(x[qi][:,None,:]*x[ci],axis=-1)
        if loss_kind=='margin':
            positive=mx.sum(mx.where(pos,sim,0),axis=1)/mx.sum(pos,axis=1)
            negative=mx.max(mx.where(judged & ~pos,sim,-1e9),axis=1)
            return mx.mean(mx.maximum(0,.2-positive+negative))
        logits=sim/cfg['temperature']
        return mx.mean(mx.logsumexp(mx.where(judged,logits,-1e9),axis=1)-mx.logsumexp(mx.where(pos,logits,-1e9),axis=1))
    vg=mx.value_and_grad(loss); optimizer=optim.AdamW(learning_rate=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    best=-1;patience=0;history=[];start=time.perf_counter();bestp=None
    dev=[r for r in read('dev') if r['query_kind']=='semantic-only']
    for epoch in range(cfg['epochs']):
        value,grads=vg(params); grads,norm=optim.clip_grad_norm(grads,cfg['clip_norm']);params=optimizer.apply_gradients(grads,params);mx.eval(params,value,optimizer.state)
        p={k:np.array(v) for k,v in params.items()}
        from training.evaluate import ndcg
        quality=np.mean([ndcg(r,list(np.argsort(-(candidate_vectors(p,r['candidates'],family)@encode_numpy(p,[r['query']],family)[0])))) for r in dev])
        history.append({'epoch':epoch+1,'loss':float(value),'devSemanticNdcg5BeforeCutoff':float(quality)})
        if quality>best+1e-9: best=float(quality);bestp=p;patience=0
        else: patience+=1
        if patience>=cfg['patience']: break
    elapsed=time.perf_counter()-start
    name=f'{architecture}-{dimension}-{family}-{loss_kind}-seed{seed}';out=Path('runs')/name;out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'weights.npz',**bestp)
    meta={'architecture':architecture,'dimension':dimension,'family':family,'loss':loss_kind,'margin':.2 if loss_kind=='margin' else None,'seed':seed,'configuration':cfg,'actualBatchSize':len(rows),'bestDevSemanticNdcg5BeforeCutoff':best,'epochs':len(history),'trainingSeconds':elapsed,'history':history,'framework':'mlx 0.31.1','device':str(mx.default_device()),'platform':platform.platform(),'dataset':'agent-authored synthetic; no reviewed judgments','status':'experimental checkpoint; not selected for deployment'}
    (out/'training.json').write_text(json.dumps(meta,indent=2)+'\n');print(name,round(best,4),round(elapsed,3),flush=True)
    return name

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/base.yaml');a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text())
    names=[]
    for arch in cfg['architectures']:
        for seed in cfg['seeds']: names.append(train_one(cfg,arch,seed))
    for seed in cfg['seeds']:
        names.append(train_one(cfg,'pooled',seed,loss_kind='margin'))
        names.append(train_one(cfg,'projected',seed,dimension=24))
        for family in ['word','char']: names.append(train_one(cfg,'pooled',seed,family=family))
    Path('eval/experiments.json').write_text(json.dumps(names,indent=2)+'\n')
if __name__=='__main__':main()
