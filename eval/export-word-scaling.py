"""Export the development-selected .85 scale and recheck exact serialized parameters."""
import importlib
import json
import struct
from pathlib import Path
import numpy as np
from training.evaluate_expanded import load_artifact

source=Path('packages/model/experiments/navigation-align0p5-seed29')
target=Path('packages/model/experiments/navigation-align0p5-seed29-word085')
params,manifest=load_artifact(source)
manifest['modelId']='navigation-align0p5-seed29-word085'
word=manifest['tensors'][0]
assert word['name']=='word'
scale=np.float32(np.float32(word['scale'])*np.float32(.85))
word['scale']=float(scale)
word['scaleF32LE']=struct.pack('<f',scale).hex()
target.mkdir(exist_ok=True)
(target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(target/'weights.bin').write_bytes((source/'weights.bin').read_bytes())
assert (target/'weights.bin').stat().st_size==32768
module=importlib.import_module('eval.typo-word-scaling-followup')
module.ARTIFACT=target
module.RATIOS=[1]
module.OUTPUT=Path('eval/typo-word-scaling-export-verification.json')
module.main()
report=json.loads(module.OUTPUT.read_text())
assert report['results'][0]['allGatesPassed']
report['stage']='Exact exported .85 word-scale artifact verification only; no new ratio selection.'
report['manifestSha256']=__import__('hashlib').sha256((target/'manifest.json').read_bytes()).hexdigest()
report['parentArtifact']=str(source)
report['exportedWordScaleRatio']=.85
report['method']='Int8 codes remain byte-identical; word dequantization scale multiplied by .85 and rounded to float32 in manifest. Reloaded exported manifest and bytes for all metrics. Expanded calibration alone fits cutoff; no holdout/diagnostic/custom-query reads.'
module.OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
