"""Controlled, same-size offline teacher distillation; no test-set access."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from training.train_expanded import read, validate_rows, SparseCache, Menus, Development
from training.evaluate_expanded import rankings, summarize, calibrate, Evaluator, load_artifact, bootstrap
from training.export import export


def teacher_reference(folder):
    cal = read('data/expanded/calibration.jsonl')
    dev = read('data/expanded/dev.jsonl')
    evaluator = Evaluator(cal, dev)
    cs = np.load(folder / 'calibration-scores.npy')
    ds = np.load(folder / 'dev-scores.npy')
    cs = [s[:len(r['candidates'])].tolist() for r, s in zip(cal, cs)]
    ds = [s[:len(r['candidates'])].tolist() for r, s in zip(dev, ds)]
    cutoff = calibrate(cal, cs)
    ranked = rankings(dev, ds, cutoff, evaluator.tiers)
    raw = rankings(dev, ds, -np.inf, evaluator.tiers, raw=True)
    return {'cutoff': cutoff, 'overall': summarize(dev, ranked, evaluator.tiers),
            'semantic': summarize([dev[i] for i in evaluator.semantic], [ranked[i] for i in evaluator.semantic]),
            'rawSemantic': summarize([dev[i] for i in evaluator.semantic], [raw[i] for i in evaluator.semantic]),
            'status': 'Offline reference only, never shipped; calibration fitted independently.'}


def train(rows, teacher, development, cfg, seed, weight, objective='mse'):
    name = f'distill-minilm-{objective}{weight}-seed{seed}'
    out = Path('runs') / name
    out.mkdir(exist_ok=False)
    validate_rows(rows)
    cache = SparseCache([t for r in rows for t in [r['query'], *[c['label'] for c in r['candidates']]]])
    menus = Menus(rows, cache)
    teacher = mx.array(teacher)
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    params = {name: mx.random.normal((1024, 16)) * .1 for name in ['word', 'char']}
    optimizer = optim.AdamW(learning_rate=cfg['learningRate'], weight_decay=cfg['weightDecay'])
    def loss(p, unique, qi, ci, positive, eligible, target):
        vectors = cache.encode(p, unique)
        cosine = mx.sum(vectors[qi, None, :] * vectors[ci], axis=-1)
        logits = cosine / cfg['temperature']
        has_positive = mx.any(positive, axis=1)
        # An arbitrary eligible placeholder prevents undefined all-masked numerators;
        # rows without positives contribute zero supervised loss, never false positives.
        numerator_mask = mx.where(has_positive[:, None], positive, eligible)
        ce = mx.logsumexp(mx.where(eligible, logits, -1e9), axis=1) - mx.logsumexp(mx.where(numerator_mask, logits, -1e9), axis=1)
        ce = mx.sum(mx.where(has_positive, ce, 0)) / mx.maximum(mx.sum(has_positive), 1)
        difference = cosine - target
        if objective == 'centered':
            # Relational distances without copying the teacher's absolute cosine offset.
            mean = mx.sum(mx.where(eligible, difference, 0), axis=1, keepdims=True) / mx.maximum(mx.sum(eligible, axis=1, keepdims=True), 1)
            difference = difference - mean
        mse = mx.sum(mx.where(eligible, difference ** 2, 0)) / mx.maximum(mx.sum(eligible), 1)
        return ce + weight * mse
    value_grad = mx.value_and_grad(loss)
    history = []
    best = -1.
    start = time.perf_counter()
    for epoch in range(1, cfg['epochs'] + 1):
        order = rng.permutation(len(rows))
        losses = []
        for offset in range(0, len(rows), cfg['batchSize']):
            indices = order[offset:offset + cfg['batchSize']]
            value, gradients = value_grad(params, *menus.batch(indices), teacher[mx.array(indices)])
            gradients, _ = optim.clip_grad_norm(gradients, cfg['clipNorm'])
            params = optimizer.apply_gradients(gradients, params)
            mx.eval(value, params, optimizer.state)
            losses.append(float(value))
        metrics = development.measure(params)
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), **metrics})
        if metrics['devNoMatchSemanticRate'] <= .05 and metrics['devSemanticNdcg5'] > best:
            best = metrics['devSemanticNdcg5']
            best_epoch = epoch
            np.savez(out / 'weights.npz', **{k: np.array(v) for k, v in params.items()})
        if epoch in [1, 10, cfg['epochs']]:
            print(name, epoch, metrics['devSemanticNdcg5'], metrics['devNoMatchSemanticRate'], flush=True)
    metadata = {'architecture': 'pooled', 'dimension': 16, 'family': 'both', 'seed': seed,
                'configuration': cfg, 'distillationWeight': weight, 'objective': objective, 'history': history,
                'bestEpoch': best_epoch if best >= 0 else None, 'trainingSeconds': time.perf_counter() - start,
                'supervision': 'Only frozen expanded train rows and train-only MiniLM cosine targets.',
                'status': 'Development-selected experiment; no reviewed final-test evidence.'}
    (out / 'training.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if best >= 0:
        artifact = Path('packages/model/experiments') / name
        manifest = export(out, 8, artifact)
        evaluator = Evaluator(read('data/expanded/calibration.jsonl'), read('data/expanded/dev.jsonl'))
        params, _ = load_artifact(artifact)
        result = evaluator.evaluate(params)
        return {'name': name, 'artifact': str(artifact), 'bestEpoch': best_epoch, 'seed': seed,
                'weight': weight, 'payloadBytes': manifest['payloadBytes'], 'sha256': manifest['payloadSha256'], **result}
    return {'name': name, 'seed': seed, 'weight': weight, 'status': 'No epoch met development no-match budget'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', default='runs/teacher-minilm')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--weights', type=float, nargs='+', default=[0, 1, 4])
    parser.add_argument('--objective', choices=['mse', 'centered'], default='mse')
    args = parser.parse_args()
    cfg = json.loads(Path('configs/experiments-distillation.json').read_text())
    folder = Path(args.teacher)
    manifest = json.loads((folder / 'manifest.json').read_text())
    for path, digest in manifest['sourceHashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Teacher target source mismatch')
    target_file = folder / 'train-scores.npy'
    if hashlib.sha256(target_file.read_bytes()).hexdigest() != manifest['scoreHashes']['train-scores.npy']:
        raise ValueError('Teacher target bytes changed')
    rows = read('data/expanded/train.jsonl')
    target = np.load(target_file)
    if target.shape != (len(rows), max(len(row['candidates']) for row in rows)) or any(not np.isfinite(target[i, :len(row['candidates'])]).all() for i, row in enumerate(rows)):
        raise ValueError('Invalid teacher targets')
    # Padding is not a candidate and must never introduce infinity into the loss.
    target = np.where(np.isfinite(target), target, 0).astype(np.float32)
    development = Development(read('data/expanded/dev.jsonl'), read('data/expanded/calibration.jsonl'))
    report_path = Path('eval/distillation-experiments.json')
    report = json.loads(report_path.read_text()) if report_path.exists() else {'configuration': cfg, 'teacher': manifest, 'reference': teacher_reference(folder), 'runs': []}
    for weight in args.weights:
        run = train(rows, target, development, cfg, args.seed, weight, args.objective)
        report['runs'].append(run)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(run), flush=True)


if __name__ == '__main__':
    main()
