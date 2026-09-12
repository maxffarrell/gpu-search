import argparse, json, hashlib
import yaml
from pathlib import Path
from training.features import features, normalize, tokens
# Original agent-authored phrases. Each domain/menu lineage stays in one split.
DOMAINS = {
 'train': [
 ('workspace', [('Members',['coworkers','people on my team','team directory']),('Billing',['payments','payment history','money owed']),('Profile',['my information','personal details','about me']),('API Keys',['developer credentials','access tokens','integration secrets']),('Notifications',['alerts','messages to me','email notices']),('Security',['protect my login','sign in protection','password settings'])]),
 ('files', [('Upload',['send files','add a document','put files online']),('Download',['save locally','get a copy','retrieve files']),('Archive',['store for later','hide old items','retire documents']),('Delete',['remove forever','erase a file','discard item']),('Sharing',['let others view','collaborator access','send to colleague']),('Versions',['previous revisions','older copies','change history'])]),
 ('projects', [('Create Project',['start new work','begin a project','add initiative']),('Delete Project',['remove project','erase initiative','discard project']),('Enable Automation',['turn on rules','activate workflow','start automatic tasks']),('Disable Automation',['turn off rules','deactivate workflow','stop automatic tasks']),('Import Data',['bring records in','load external records','ingest data']),('Export Data',['take records out','download records','extract data'])]),
 ],
 'dev': [
 ('commerce', [('Customers',['buyers','people who bought','client directory']),('Orders',['purchases','things sold','purchase history']),('Refunds',['give money back','return payment','reverse purchase']),('Inventory',['stock levels','available goods','items on hand']),('Discounts',['price reductions','coupons','promotion codes']),('Shipping',['deliveries','send parcels','delivery settings'])]),
 ('calendar', [('Events',['scheduled occasions','appointments','upcoming meetings']),('Availability',['free time','open slots','when I can meet']),('Reminders',['notify before meeting','appointment alerts','upcoming notices']),('Time Zone',['local clock region','regional time','clock location']),('Invite Guests',['add attendees','ask people to join','bring participants']),('Cancel Meeting',['call off appointment','remove scheduled gathering','stop meeting'])]),
 ],
 'test': [
 ('publishing', [('Drafts',['unfinished writing','unpublished articles','work in progress']),('Published',['live articles','visible posts','released stories']),('Comments',['reader replies','audience feedback','discussion']),('Media Library',['pictures and videos','uploaded images','visual assets']),('Authors',['writers','content creators','people who write']),('Analytics',['reader statistics','traffic reports','audience numbers'])])]
}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/data.yaml'); args=p.parse_args(); config=yaml.safe_load(Path(args.config).read_text())
    root=Path(config['output']); root.mkdir(exist_ok=True)
    rows_by_split={}
    for split, domains in DOMAINS.items():
        rows=[]
        for family, concepts in domains:
            candidates=[{'id':str(i),'label':label} for i,(label,_) in enumerate(concepts)]
            for i,(label,phrases) in enumerate(concepts):
                examples=[(label,'exact'),(label[:-1]+label[-1]*2,'typo')]+[(q,'semantic-only') for q in phrases]
                for j,(query,kind) in enumerate(examples):
                    rows.append(dict(id=f'{family}-{i}-{j}',product_family=family,source_id='synthetic-v1',query=query,candidates=candidates,relevance={str(k):3 if i==k else 0 for k in range(len(concepts))},judgment_status='synthetic-unreviewed',query_kind=kind,augmentation_parent=f'{family}-{i}',split_group=family))
            for i,query in enumerate(['xqzv','purple dinosaur','ocean temperature','banana bread recipe','lunar spacecraft','zzzxq','dancing penguin','volcano depth','zebra habitat','solar eclipse']):
                # Unique wording per split avoids literal query overlap; still synthetic easy negatives.
                rows.append(dict(id=f'{family}-none-{i}',product_family=family,source_id='synthetic-v1',query=query+' '+family,candidates=candidates,relevance={str(k):0 for k in range(len(concepts))},judgment_status='synthetic-unreviewed',query_kind='no-match',augmentation_parent=None,split_group=family))
        rows_by_split[split]=rows
        (root/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    aliases={label:qs for _,concepts in DOMAINS['train'] for label,qs in concepts}
    (root/'aliases.json').write_text(json.dumps(aliases,indent=2)+'\n')
    queries={s:{normalize(r['query']) for r in rs} for s,rs in rows_by_split.items()}
    audit={'status':'synthetic feasibility only; not reviewed or sealed generalization evidence','counts':{s:len(rs) for s,rs in rows_by_split.items()},'reviewedCount':0,'families':{s:[f for f,_ in ds] for s,ds in DOMAINS.items()},'queryOverlap':{f'{a}/{b}':sorted(queries[a]&queries[b]) for a,b in [('train','dev'),('train','test'),('dev','test')]},'encoderTruncations':{s:sum(features(r['query'])['truncated'] for r in rs) for s,rs in rows_by_split.items()}}
    (root/'leakage-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (root/'sources.json').write_text(json.dumps({'id':'synthetic-v1','type':'agent-authored synthetic','license':'MIT (original project fixtures)','reviewerStatus':'unreviewed','externalCorpora':[],'teacher':None,'script':'training/prepare.py','scriptSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'warning':'Concepts and relevance are illustrative guesses, not human judgments. No product-quality claim.'},indent=2)+'\n')
    fixture_texts=['\u001cHello\u001fWorld', '','Profile','myProfile','C++ C# C','ＡＰＩ Keys','café','a_a_a','𐐀 test','\u2003Hello\tWorld','x'*65,' '.join(['repeated']*40)]
    Path('fixtures/features.json').write_text(json.dumps([{'text':t,'normalized':normalize(t),'tokens':tokens(t),**features(t)} for t in fixture_texts],indent=2)+'\n')
    print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
