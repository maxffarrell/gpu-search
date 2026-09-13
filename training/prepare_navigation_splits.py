"""Provider-separated positive-only navigation corpus; no fabricated negatives."""
import hashlib
import json
from pathlib import Path
from training.features import normalize


def prepare(output='data/navigation-splits'):
    source=Path('data/navigation/panels.json')
    panels=json.loads(source.read_text())
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    partitions={'train':'kde','dev':'xfce','holdout':'gnome'}
    result={}
    for partition,provider in partitions.items():
        selected=[p for p in panels if p['provider']==provider and p['entry_role'] in ['settings-panel','settings-module']]
        candidates=[{'id':p['id'],'label':p['label']} for p in selected]
        groups={}
        for panel in selected:
            for query in panel['keywords']:
                normalized=normalize(query)
                if not 3<=len(normalized)<=256:continue
                groups.setdefault(normalized,{'query':query,'positive':set()})['positive'].add(panel['id'])
        rows=[]
        for i,(normalized,group) in enumerate(sorted(groups.items())):
            rows.append({'id':f'navigation-{provider}-{i:04d}','query':group['query'],
                         'candidates':candidates,'relevance':{key:3 for key in sorted(group['positive'])},
                         'source_id':provider,'evaluation_slice':f'navigation-{provider}',
                         'judgment_status':'upstream author search keywords; unlisted destinations unjudged',
                         'query_normalized':normalized})
        file=out/f'{partition}.jsonl'
        file.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
        result[partition]={'provider':provider,'destinations':len(candidates),'queries':len(rows),
                           'multiPositiveQueries':sum(len(r['relevance'])>1 for r in rows),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    manifest={'source':str(source),'sourceSha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'partitions':result,'training':'KDE positive-alignment only; never convert unlisted candidates into negatives.',
              'selection':'XFCE positive retrieval and existing expanded-development regression limits.',
              'calibration':'Existing expanded calibration only; navigation corpus has no judged no-match rows.',
              'holdout':'GNOME provider; do not score until model selection is frozen. Cross-provider wording overlap will be reported.',
              'qualityScope':'Known-positive retrieval lower bound, not fully reviewed relevance/no-match evidence.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':prepare()
