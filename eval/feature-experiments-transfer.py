"""One post-selection pilot transfer. Selection must already be frozen."""
import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from training.features_bigram import encode_numpy,candidate_vectors
from training.evaluate_expanded import read_rows,summarize,rankings,bootstrap,load_artifact,Evaluator
from training.evaluate import lexical
frozen=json.loads(Path('eval/feature-experiments-selection.json').read_text())
folder=Path('packages/model/experiments')/frozen['selected'];manifest=json.loads((folder/'manifest.json').read_text());blob=(folder/'weights.bin').read_bytes()
assert hashlib.sha256(blob).hexdigest()==frozen['payloadSha256']
params={t['name']:np.frombuffer(blob[t['byteOffset']:t['byteOffset']+t['byteLength']],np.int8).reshape(t['shape']).astype(np.float32)*np.float32(t['scale']) for t in manifest['tensors']}
rows=read_rows('data/dev.jsonl');cal=read_rows('data/expanded/calibration.jsonl');lex=[[lexical(r['query'],c) for c in r['candidates']] for r in rows]
scores=[(candidate_vectors(params,r['candidates'])@encode_numpy(params,[r['query']])[0]).tolist() for r in rows]
cutoff=manifest.get('calibratedCutoff',manifest.get('quantizedMetrics',{}).get('cutoff'))
raw=rankings(rows,scores,-1,lex,raw=True);combined=rankings(rows,scores,cutoff,lex)
systems={'bigram-int8-raw':summarize(rows,raw,None),'bigram-int8-combined':summarize(rows,combined,lex)}
old,m=load_artifact('packages/model/experimental');ev=Evaluator(cal,rows);ss=ev.scores(old)[len(cal):];old_raw=rankings(rows,ss,-1,lex,raw=True);systems['original-pilot-int8-raw']=summarize(rows,old_raw,None)
positive=[i for i,r in enumerate(rows) if any(v>=2 for v in r['relevance'].values())]
actual=systems['bigram-int8-raw']['ndcg5'];guards={'originalPilotReference':.6548,'deployedReference':.6460,'maximumRegression':.01,'vsOriginalPilotPass':actual>=.6548-.01,'vsDeployedPass':actual>=.6460-.01}
result={'selectionFrozenBeforeTransfer':frozen,'scope':'Original pilot development opened only after expanded-dev selection; no retuning against this report','systems':systems,'pairedRawComparisonToOriginalPilot':bootstrap(rows,raw,old_raw,positive),'promotionGuards':guards}
Path('eval/feature-experiments-transfer.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
