"""Sparse MLX weak-label experiment, preserving original pilot artifacts.

Uses the frozen family-mean feature representation with repeated hashed features.
No development text is used as a training target or negative. All inputs in this
experiment must have label-only candidates; aliases/context are rejected loudly.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import yaml

from training.features import features, normalize
from training.evaluate import lexical, ndcg


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_rows(rows):
    for row in rows:
        if not row['candidates']:
            raise ValueError('Training menus cannot be empty')
        for candidate in row['candidates']:
            if candidate.get('aliases') or candidate.get('context'):
                raise ValueError('Expanded experiment is label-only; metadata composition requires a separately specified experiment')


class SparseCache:
    def __init__(self, texts):
        self.texts = list(dict.fromkeys(texts))
        self.lookup = {text: i for i, text in enumerate(self.texts)}
        fs = [features(t) for t in self.texts]
        self.truncated = sum(f['truncated'] for f in fs)
        self.arrays = []
        for name, bound in [('wordIds', 32), ('charIds', 512)]:
            width = max(1, max((len(f[name]) for f in fs), default=0))
            if width > bound:
                raise ValueError('Feature contract overflow')
            ids = np.zeros((len(fs), width), np.int32)
            weights = np.zeros((len(fs), width), np.float32)
            for i, f in enumerate(fs):
                n = len(f[name])
                ids[i, :n] = f[name]
                if n:
                    weights[i, :n] = 1 / n
            self.arrays.extend([mx.array(ids), mx.array(weights)])
        mx.eval(*self.arrays)

    def encode(self, params, ids):
        wi, ww, ci, cw = self.arrays
        w = mx.sum(params['word'][wi[ids]] * ww[ids, :, None], axis=1)
        c = mx.sum(params['char'][ci[ids]] * cw[ids, :, None], axis=1)
        v = .5 * (w + c)
        if 'W1' in params:
            v = mx.tanh(v @ params['W1'].T + params['b1']) @ params['W2'].T + params['b2']
        return v / mx.maximum(mx.sqrt(mx.sum(v*v, axis=-1, keepdims=True)), 1e-8)

    def all_vectors(self, params, chunk=512):
        result = []
        for start in range(0, len(self.texts), chunk):
            v = self.encode(params, mx.arange(start, min(start+chunk, len(self.texts))))
            mx.eval(v)
            result.append(np.array(v))
        return np.concatenate(result)


class Menus:
    def __init__(self, rows, cache):
        self.rows = rows
        self.q = np.array([cache.lookup[r['query']] for r in rows], np.int32)
        width = max(len(r['candidates']) for r in rows)
        self.c = np.zeros((len(rows), width), np.int32)
        self.pos = np.zeros_like(self.c, bool)
        self.judged = np.zeros_like(self.c, bool)
        for i, r in enumerate(rows):
            for j, c in enumerate(r['candidates']):
                self.c[i,j] = cache.lookup[c['label']]
                self.pos[i,j] = r['relevance'].get(c['id'], -1) >= 2
                self.judged[i,j] = c['id'] in r['relevance']

    def batch(self, ids):
        together = np.concatenate([self.q[ids,None], self.c[ids]], axis=1)
        unique, reverse = np.unique(together, return_inverse=True)
        reverse = reverse.reshape(together.shape)
        return tuple(map(mx.array, [unique, reverse[:,0], reverse[:,1:], self.pos[ids], self.judged[ids]]))


class Development:
    def __init__(self, rows, calibration):
        self.rows = rows
        self.calibration = calibration
        all_rows = rows + calibration
        self.cache = SparseCache([t for r in all_rows for t in [r['query'], *[c['label'] for c in r['candidates']]]])
        self.dev = Menus(rows, self.cache)
        self.cal = Menus(calibration, self.cache)
        self.lex = [[lexical(r['query'],c) for c in r['candidates']] for r in rows]
        self.relevant = [i for i,r in enumerate(rows) if any(g>=2 for g in r['relevance'].values())]
        self.semantic = [i for i in self.relevant if not any(k[0] and rows[i]['relevance'].get(c['id'],0)>=2 for k,c in zip(self.lex[i],rows[i]['candidates']))]
        self.no_match = [i for i,r in enumerate(rows) if r['relevance'] and max(r['relevance'].values())==0]

    def measure(self, params):
        vectors = self.cache.all_vectors(params)
        scores = np.sum(vectors[self.dev.q,None,:] * vectors[self.dev.c], axis=-1)
        scores = np.where((np.linalg.norm(vectors[self.dev.q],axis=-1)[:,None]>=1e-8) & (np.linalg.norm(vectors[self.dev.c],axis=-1)>=1e-8),scores,-np.inf)
        cs = np.sum(vectors[self.cal.q,None,:] * vectors[self.cal.c], axis=-1)
        cs = np.where((np.linalg.norm(vectors[self.cal.q],axis=-1)[:,None]>=1e-8) & (np.linalg.norm(vectors[self.cal.c],axis=-1)>=1e-8),cs,-np.inf)
        maxima = sorted([float(max(s[:len(r['candidates'])])) for r,s in zip(self.calibration, cs) if r['relevance'] and max(r['relevance'].values())==0], reverse=True)
        if not maxima:
            raise ValueError('Separate calibration set must contain no-match menus')
        permitted = int(.05*len(maxima))
        cutoff = float(np.nextafter(np.float32(maxima[permitted]), np.float32(np.inf))) if np.isfinite(maxima[permitted]) else 1.01
        ranks = []
        semantic_return = []
        for i,r in enumerate(self.rows):
            keys = []
            semantic_return.append(False)
            for j,c in enumerate(r['candidates']):
                k = self.lex[i][j]
                if not k[0] and len(normalize(r['query']))>=3 and scores[i,j]>=cutoff:
                    k = (1,0,float(scores[i,j])); semantic_return[-1] = True
                if k[0]: keys.append((k,j))
            ranks.append([j for k,j in sorted(keys,key=lambda pair:(*[-v for v in pair[0]],pair[1]))])
        def mean(items): return float(np.mean(items)) if items else None
        return {'cutoff':cutoff,'calibrationNoMatchCount':len(maxima),'calibrationNoMatchSemanticRate':sum(s>=cutoff for s in maxima)/len(maxima),'devSemanticCount':len(self.semantic),'devSemanticNdcg5':mean([ndcg(self.rows[i],ranks[i]) for i in self.semantic]),'devSemanticCoverage':mean([bool(ranks[i]) for i in self.semantic]),'devRelevantCoverage':mean([bool(ranks[i]) for i in self.relevant]),'devOverallNdcg5':mean([ndcg(self.rows[i],ranks[i]) for i in self.relevant]),'devNoMatchSemanticRate':mean([semantic_return[i] for i in self.no_match]),'devNoMatchAnyRate':mean([bool(ranks[i]) for i in self.no_match])}


def train_one(cfg, rows, development, seed, arm, architecture='pooled'):
    validate_rows(rows)
    supervised = [r for r in rows if any(g>=2 for g in r['relevance'].values())]
    cache = SparseCache([t for r in supervised for t in [r['query'],*[c['label'] for c in r['candidates']]]])
    menus = Menus(supervised,cache)
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    dim = cfg['dimension']
    params = {'word':mx.random.normal((1024,dim))*.1,'char':mx.random.normal((1024,dim))*.1}
    if architecture == 'projected':
        params.update(W1=mx.random.normal((24,dim))*.2,b1=mx.zeros(24),W2=mx.random.normal((dim,24))*.2,b2=mx.zeros(dim))
    opt = optim.AdamW(learning_rate=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    def loss(p, unique, qi, ci, positive, eligible):
        v = cache.encode(p,unique)
        sim = mx.sum(v[qi,None,:]*v[ci],axis=-1)/cfg['temperature']
        return mx.mean(mx.logsumexp(mx.where(eligible,sim,-1e9),axis=1)-mx.logsumexp(mx.where(positive,sim,-1e9),axis=1))
    vg = mx.value_and_grad(loss)
    name = f'expanded-{arm}-{architecture}-seed{seed}' + (('-'+cfg['run_tag']) if cfg.get('run_tag') else '')
    out = Path('runs')/name
    if out.exists():
        raise FileExistsError(f'Preserving immutable prior experiment: {out}')
    out.mkdir(parents=True)
    best_key = (-1.,-1.); best_epoch = 0; stale = 0; history = []; start = time.perf_counter(); optimizer_seconds = 0.
    for epoch in range(1,cfg['epochs']+1):
        epoch_examples = cfg.get('epoch_examples', len(supervised))
        order = np.concatenate([rng.permutation(len(supervised)) for _ in range((epoch_examples+len(supervised)-1)//len(supervised))])[:epoch_examples]
        losses=[]; train_start=time.perf_counter()
        for offset in range(0,len(order),cfg['batch_size']):
            batch = menus.batch(order[offset:offset+cfg['batch_size']])
            value, gradients = vg(params,*batch)
            gradients,_ = optim.clip_grad_norm(gradients,cfg['clip_norm'])
            params = opt.apply_gradients(gradients,params)
            mx.eval(value,params,opt.state)
            losses.append(float(value))
        optimizer_seconds += time.perf_counter()-train_start
        metrics = development.measure(params)
        checkpoint = {k:np.array(v) for k,v in params.items()}
        if epoch in (1,10):
            cp=out/f'epoch-{epoch:03d}';cp.mkdir();np.savez(cp/'weights.npz',**checkpoint)
            (cp/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        key = (metrics['devSemanticNdcg5'],metrics['devRelevantCoverage'])
        if key > best_key:
            best_key=key;best_epoch=epoch;stale=0;np.savez(out/'weights.npz',**checkpoint);best_metrics=metrics
        else: stale+=1
        history.append({'epoch':epoch,'optimizerSteps':epoch*((cfg.get('epoch_examples',len(supervised))+cfg['batch_size']-1)//cfg['batch_size']),'meanBatchLoss':float(np.mean(losses)),**metrics,'elapsedSeconds':time.perf_counter()-start})
        print(name,epoch,round(metrics['devSemanticNdcg5'],4),round(metrics['devSemanticCoverage'],4),flush=True)
        if epoch>=cfg['minimum_epochs'] and stale>=cfg['patience']:
            break
    metadata={'architecture':architecture,'dimension':dim,'family':'both','seed':seed,'dataArm':arm,'configuration':cfg,'trainRecords':len(rows),'supervisedRecords':len(supervised),'noPositiveRecordsExcluded':len(rows)-len(supervised),'examplesPerEpoch':cfg.get('epoch_examples',len(supervised)),'optimizerStepsPerEpoch':(cfg.get('epoch_examples',len(supervised))+cfg['batch_size']-1)//cfg['batch_size'],'uniqueTrainingTexts':len(cache.texts),'truncatedTrainingTexts':cache.truncated,'bestEpoch':best_epoch,'epochs':len(history),'totalOptimizerSteps':len(history)*((cfg.get('epoch_examples',len(supervised))+cfg['batch_size']-1)//cfg['batch_size']),'selectedOptimizerSteps':best_epoch*((cfg.get('epoch_examples',len(supervised))+cfg['batch_size']-1)//cfg['batch_size']),'bestMetrics':best_metrics,'history':history,'trainingSeconds':time.perf_counter()-start,'optimizerSeconds':optimizer_seconds,'framework':'mlx 0.31.1','device':str(mx.default_device()),'platform':platform.platform(),'status':'weak-label experimental selection; no reviewed final-test evidence','selection':'lexicographic post-cutoff development semantic nDCG@5 then relevant-query coverage; independent no-match calibration; earliest epoch wins exact ties'}
    (out/'training.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for epoch in (1,10):
        cp=out/f'epoch-{epoch:03d}'
        if cp.exists():
            (cp/'training.json').write_text(json.dumps({**{k:v for k,v in metadata.items() if k!='history'},'checkpointEpoch':epoch,'status':'fixed epoch comparison; not selected'},indent=2)+'\n')
    return {'checkpoint':str(out),'seed':seed,'dataArm':arm,'architecture':architecture,'bestEpoch':best_epoch,'epochs':len(history),'trainingSeconds':metadata['trainingSeconds'],'optimizerSeconds':optimizer_seconds,**best_metrics}


def decorate_report(report):
    report['timingScope'] = 'Materialized training loop includes optimizer updates, per-epoch development scoring and checkpoint writes; excludes data loading, feature preparation and initial lexical evaluation cache construction. OptimizerSeconds isolates minibatch work.'
    report['selectionCaveat'] = 'Calibration constrains cutoff to <=5% no-match on calibration only; development rates are measured independently. Artifact promotion must consider development false positives as well as nDCG, rather than taking the highest nDCG unconditionally.'
    report['fixedEpochComparison'] = []
    hashes = {}
    for run in report['runs']:
        folder = Path(run['checkpoint'])
        meta = json.loads((folder/'training.json').read_text())
        report['fixedEpochComparison'].append({'checkpoint':str(folder),'seed':meta['seed'],'dataArm':meta['dataArm'],'optimizerStepsPerEpoch':meta['optimizerStepsPerEpoch'],'selectedEpoch':meta['bestEpoch'],'selectedOptimizerSteps':meta['selectedOptimizerSteps'],'totalOptimizerSteps':meta['totalOptimizerSteps'],'points':[{'epoch':e['epoch'],'optimizerSteps':e['optimizerSteps'],'semanticNdcg5':e['devSemanticNdcg5'],'semanticCoverage':e['devSemanticCoverage'],'devNoMatchSemanticRate':e['devNoMatchSemanticRate']} for e in meta['history'] if e['epoch'] in (1,10,meta['bestEpoch'])]})
        for path in sorted(folder.glob('**/weights.npz')):
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    report['checkpointHashes'] = hashes
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/expanded.yaml');parser.add_argument('--arm',choices=['all','more-data','old-data-control'],default='all');parser.add_argument('--seed',type=int);parser.add_argument('--tag',help='Optional new experiment suffix; existing checkpoint directories are never overwritten');args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text());folder=Path(cfg['data_dir'])
    if args.tag:
        if not args.tag.replace('-','').replace('_','').isalnum():
            parser.error('--tag must contain only letters, digits, hyphens or underscores')
        cfg['run_tag']=args.tag
    source_paths=[folder/'train.jsonl',folder/'dev.jsonl',folder/'calibration.jsonl',Path(cfg['control_data'])]
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    cfg['source_hashes']=source_hashes
    dev=read(folder/'dev.jsonl');cal=read(folder/'calibration.jsonl');validate_rows(dev+cal)
    development=Development(dev,cal)
    cfg['epoch_examples']=sum(any(g>=2 for g in r['relevance'].values()) for r in read(folder/'train.jsonl'))
    results=[]
    for arm in ['old-data-control','more-data'] if args.arm=='all' else [args.arm]:
        rows=read(cfg['control_data'] if arm=='old-data-control' else folder/'train.jsonl')
        for architecture in cfg['architectures']:
            for seed in [args.seed] if args.seed is not None else cfg['seeds']:
                results.append(train_one(cfg,rows,development,seed,arm,architecture))
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=source_hashes[str(p)] for p in source_paths):
        raise RuntimeError('Source changed during training: reject mixed-data aggregate')
    out=Path('eval/expanded-training.json')
    prior=json.loads(out.read_text())['runs'] if out.exists() else []
    by_path={r['checkpoint']:r for r in prior+results}
    report = {'status':'experimental weak-label development evidence only','configuration':cfg,'sourceHashes':source_hashes,'control':'Identical sparse features, optimizer, seeds, optimizer updates per epoch, development selection and calibration; only unique training records differ. The old-data control repeats shuffled original training rows to match the expanded positive-example count per epoch. Early stopping can select different total update counts. Fixed epoch-1/10 artifacts compare equal update budgets.','runs':list(by_path.values())}
    out.write_text(json.dumps(decorate_report(report),indent=2)+'\n')

if __name__=='__main__':main()
