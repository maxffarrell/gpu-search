"""Regenerate numerical parity evidence from the selected int8 bytes with NumPy."""
import json
import hashlib
from pathlib import Path
from training.evaluate_expanded import load_artifact
from training.model import encode_numpy, candidate_vectors
artifact = Path('packages/model/experiments/navigation-align0p5-seed29-word085')
params, manifest = load_artifact(artifact)
texts = ['Profile', 'coworkers', 'network preferences', 'dark appearance', 'C++', 'myProfile', '音量', '', 'wireless connections']
candidates = [
    {'id': 'network', 'label': 'Network', 'aliases': ['Wi-Fi', 'Connections'], 'context': 'System preferences'},
    {'id': 'appearance', 'label': 'Appearance', 'aliases': ['Theme'], 'context': 'Display settings'},
    {'id': 'profile', 'label': 'Profile'},
]
vectors = encode_numpy(params, texts)
scores = vectors @ candidate_vectors(params, candidates).T
Path('fixtures/navigation-word085-model.json').write_text(json.dumps({'manifestSha256':hashlib.sha256((artifact/'manifest.json').read_bytes()).hexdigest(),'modelSha256':manifest['payloadSha256'], 'source':'NumPy encode_numpy and candidate_vectors applied to dequantized exported int8 bytes', 'candidates':candidates, 'cases':[{'text':text, 'vector':vectors[i].tolist(), 'scores':scores[i].tolist()} for i,text in enumerate(texts)]}, indent=2)+'\n')
