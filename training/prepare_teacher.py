"""Offline MiniLM reference and train-only distillation targets; never shipped.

Run in an isolated environment with torch==2.10.0, transformers==4.57.6,
numpy==2.4.3. Downloads only a pinned safetensors model (no remote code).
Evaluation embeddings never become training targets. No sealed test is opened.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
SOURCES = {'train': 'data/expanded/train.jsonl', 'calibration': 'data/expanded/calibration.jsonl',
           'dev': 'data/expanded/dev.jsonl', 'transfer': 'data/dev.jsonl'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/teacher-minilm')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = {k: [json.loads(line) for line in Path(p).read_text().splitlines() if line] for k, p in SOURCES.items()}
    texts = sorted({t for part in rows.values() for row in part
                    for t in [row['query'], *[v for c in row['candidates'] for v in [c['label'], *c.get('aliases', []), c.get('context', '')]]]})
    location = snapshot_download(MODEL, revision=REVISION,
                                 allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json',
                                                 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'README.md'])
    tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = AutoModel.from_pretrained(location, local_files_only=True, use_safetensors=True).to(device).eval()
    vectors = []
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(texts), 128):
            batch = tokenizer(texts[offset:offset + 128], padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            encoded = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vectors.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
            if offset % 1024 == 0:
                print(f'Encoded {min(offset + 128, len(texts))}/{len(texts)} texts on {device}', flush=True)
    vectors = np.concatenate(vectors).astype(np.float32)
    lookup = {text: vectors[i] for i, text in enumerate(texts)}
    def candidate_vector(candidate):
        vector = np.mean([lookup[t] for t in [candidate['label'], *candidate.get('aliases', [])]], axis=0)
        if candidate.get('context'):
            vector += .25 * lookup[candidate['context']]
        return vector / max(float(np.linalg.norm(vector)), 1e-8)
    for name, part in rows.items():
        scores = np.full((len(part), max(len(r['candidates']) for r in part)), -np.inf, np.float32)
        for i, row in enumerate(part):
            scores[i, :len(row['candidates'])] = [lookup[row['query']] @ candidate_vector(c) for c in row['candidates']]
        np.save(out / f'{name}-scores.npy', scores)
    manifest = {'model': MODEL, 'revision': REVISION, 'license': 'Apache-2.0 per pinned model card',
                'device': device, 'dimension': 384, 'uniqueTexts': len(texts), 'maxWordPieces': 256,
                'pooling': 'attention-mask mean pooling, L2 normalized, deployment-equivalent alias/context composition',
                'sourceHashes': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES.values()},
                'modelFiles': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(Path(location).glob('*')) if p.is_file()},
                'scoreHashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*-scores.npy'))},
                'encodingSeconds': time.perf_counter() - start,
                'trainingBoundary': 'Student reads only train-scores.npy; other partitions exclusively for offline evaluation.',
                'dependencies': {'torch': torch.__version__, 'transformers': '4.57.6', 'numpy': np.__version__}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
