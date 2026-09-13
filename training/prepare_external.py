"""Reproducible weak retrieval supervision from licensed upstream intent datasets.

python -m training.prepare_external [--download]
No upstream test member is accessed, emitted, scored, or used for selection.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, pathlib, random, re, unicodedata, urllib.request
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/external'
OUT = ROOT / 'data/expanded'
SOURCES = {
 'clinc150': {'repo': 'clinc/oos-eval', 'revision': '828f8093932c8fe6ca7936c3d2e52903b1c523de', 'license': 'CC-BY-3.0', 'paths': ['LICENSE', 'README.md', 'data/data_full.json']},
 'banking77': {'repo': 'PolyAI-LDN/task-specific-datasets', 'revision': '57ec275d8078af65b7731c2a98be812d844a6d6b', 'license': 'CC-BY-4.0', 'paths': ['LICENSE', 'README.md', 'banking_data/train.csv', 'banking_data/categories.json']},
}
# Group related operations BEFORE assigning partitions. Never split neighboring intent
# names independently and then describe the result as unseen-family evaluation.
CLINC_GROUPS = {
 'settings': 'change_accent change_ai_name change_language change_speed change_user_name change_volume reset_settings user_name whisper_mode',
 'calendar': 'calendar calendar_update meeting_schedule schedule_meeting',
 'reminders': 'alarm reminder reminder_update timer',
 'lists': 'shopping_list shopping_list_update todo_list todo_list_update',
 'device': 'find_phone sync_device smart_home',
 'communication': 'make_call text',
 'utilities': 'calculator date definition measurement_conversion spelling time timezone translate',
 'account_security': 'account_blocked freeze_account pin_change report_fraud',
 'payments': 'balance bill_balance bill_due pay_bill transactions transfer spending_history',
 'orders': 'order order_status cancel',
 'music': 'next_song play_music update_playlist what_song',
 'location': 'current_location directions distance share_location',
 'employment': 'pto_balance pto_request pto_request_status pto_used payday income w2',
}
# Banking is supplementary support-domain coverage, capped at 40 training records per
# known intent. Generic CLINC travel/food/chitchat are excluded altogether.
BANK_GROUPS = {
 'identity': 'why_verify_identity unable_to_verify_identity verify_my_identity verify_source_of_funds',
 'virtual_card': 'getting_virtual_card disposable_card_limits virtual_card_not_working get_disposable_virtual_card',
 'personal_security': 'edit_personal_details passcode_forgotten lost_or_stolen_phone terminate_account',
 'card_delivery': 'card_arrival card_delivery_estimate get_physical_card order_physical_card getting_spare_card',
 'card_security': 'lost_or_stolen_card compromised_card pin_blocked change_pin card_swallowed',
 'card_usage': 'card_linking card_not_working contactless_not_working card_acceptance activate_my_card card_about_to_expire visa_or_mastercard apple_pay_or_google_pay',
 'transfers': 'cancel_transfer transfer_not_received_by_recipient declined_transfer pending_transfer transfer_timing beneficiary_not_allowed transfer_fee_charged receiving_money failed_transfer transfer_into_account balance_not_updated_after_bank_transfer',
 'top_up': 'automatic_top_up top_up_by_bank_transfer_charge pending_top_up top_up_limits top_up_reverted topping_up_by_card verify_top_up top_up_by_cash_or_cheque top_up_failed top_up_by_card_charge',
 'card_payments': 'card_payment_wrong_exchange_rate extra_charge_on_statement card_payment_fee_charged card_payment_not_recognised pending_card_payment declined_card_payment transaction_charged_twice reverted_card_payment?',
 'cash': 'pending_cash_withdrawal wrong_amount_of_cash_received balance_not_updated_after_cheque_or_cash_deposit atm_support declined_cash_withdrawal wrong_exchange_rate_for_cash_withdrawal cash_withdrawal_not_recognised cash_withdrawal_charge',
 'refund': 'request_refund Refund_not_showing_up',
 'currency': 'exchange_rate fiat_currency_support exchange_via_app supported_cards_and_currencies exchange_charge',
 'eligibility': 'age_limit country_support',
}
HOLDOUT = {'clinc150:music':'dev', 'clinc150:location':'dev', 'clinc150:employment':'calibration', 'banking77:identity':'dev', 'banking77:virtual_card':'calibration'}

def digest(text: str) -> str: return hashlib.sha256(text.encode()).hexdigest()
def key(text: str) -> str:
 text = unicodedata.normalize('NFKC', text)
 return re.sub(r'\s+', ' ', text.translate(str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'))).strip()
def read_json(path): return json.loads(path.read_text())
def write_json(path, value): path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
def download():
 provenance = []
 for source, config in SOURCES.items():
  (RAW/source).mkdir(parents=True, exist_ok=True)
  for remote in config['paths']:
   url=f"https://raw.githubusercontent.com/{config['repo']}/{config['revision']}/{remote}"
   payload=urllib.request.urlopen(url, timeout=60).read()
   provenance.append({'source':source,'url':url,'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)})
   if remote.endswith('data_full.json'):
    # Upstream combines all partitions in one JSON transport. JSON decoding is
    # required, but test/oos_test values are never accessed or materialized as files.
    bundled=json.loads(payload)
    for split in ['train','val','oos_train','oos_val']:
     (RAW/source/f'{split}.json').write_text(json.dumps(bundled[split],ensure_ascii=False)+'\n')
   else: (RAW/source/pathlib.Path(remote).name).write_bytes(payload)
 write_json(RAW/'download-manifest.json', provenance)

def prepare():
 OUT.mkdir(parents=True, exist_ok=True)
 groups={source:{intent:family for family,intents in mapping.items() for intent in intents.split()} for source,mapping in [('clinc150',CLINC_GROUPS),('banking77',BANK_GROUPS)]}
 observations=[]
 for split in ['train','val']:
  for i,(query,intent) in enumerate(read_json(RAW/'clinc150'/f'{split}.json')):
   if intent in groups['clinc150']: observations.append(('clinc150',split,i,query,intent))
 with (RAW/'banking77/train.csv').open() as handle:
  for i,row in enumerate(csv.DictReader(handle)):
   intent=row['category']
   if intent in groups['banking77']: observations.append(('banking77','train',i,row['text'],intent))
 labels={s:{i:i.replace('_',' ').replace('?','').strip().capitalize() for i in group} for s,group in groups.items()}
 partitions=collections.defaultdict(list)
 dropped=collections.Counter()
 seen={}
 # Preserve original TRAIN only; do not even open original dev/test here.
 for line in (ROOT/'data/train.jsonl').read_text().splitlines():
  row=json.loads(line); normalized=key(row['query'])
  if normalized in seen: dropped['original_duplicate']+=1;continue
  seen[normalized]='train'; row={**row,'evaluation_slice':'original-train','source_split':'train'}
  partitions['train'].append(row)
 # Stable per-intent source-train cap and wording assignment. This cap is before
 # negatives are derived; no label-frequency balancing consults validation/test.
 buckets=collections.defaultdict(list)
 for obs in observations:buckets[(obs[0],obs[1],obs[4])].append(obs)
 selected=[]
 for (source,source_split,intent),rows in sorted(buckets.items()):
  family=source+':'+groups[source][intent]; hold=HOLDOUT.get(family)
  rows.sort(key=lambda row:digest('wording-v1:'+key(row[3])))
  for n,row in enumerate(rows):
   if hold:
    # CLINC unseen-intent uses official val only; its train examples stay unused.
    if source=='clinc150' and source_split!='val':dropped['held_family_upstream_train_reserved']+=1;continue
    if source=='banking77' and n>=20:dropped['held_family_cap']+=1;continue
    selected.append((hold,'unseen-intent-family',row));continue
   if source=='clinc150':
    partition='train' if source_split=='train' else ('dev' if n%2==0 else 'calibration')
   else:
    if n>=60:dropped['banking_cap']+=1;continue
    partition='train' if n<40 else ('dev' if n<50 else 'calibration')
   selected.append((partition,'known-intent-wording' if partition!='train' else 'train',row))
 # Canonical query duplicate ownership is training, then dev, then calibration.
 selected.sort(key=lambda item:({'train':0,'dev':1,'calibration':2}[item[0]],item[2][0],item[2][1],item[2][2]))
 for partition,slice_name,(source,source_split,source_row,query,intent) in selected:
  normalized=key(query)
  if not normalized or len(query)>256:dropped['empty_or_oversize_query']+=1;continue
  if normalized in seen:dropped['normalized_duplicate']+=1;continue
  seen[normalized]=partition
  family=source+':'+groups[source][intent]
  pool=[i for i in labels[source] if source+':'+groups[source][i] not in HOLDOUT or HOLDOUT[source+':'+groups[source][i]]==partition]
  # Hard negatives favor overlapping label terms; remaining distractors sampled
  # reproducibly. Source class inequality is weak evidence, not menu judgment.
  rng=random.Random(int(digest(f'{source}:{source_split}:{source_row}')[:16],16))
  other=[i for i in pool if i!=intent]
  rng.shuffle(other)
  words=set(intent.lower().split('_'))
  other.sort(key=lambda i:-len(words.intersection(i.lower().split('_'))))
  menu=[intent]+other[:9]
  no_match=int(digest('negative:'+normalized)[:8],16) % (10 if partition=='train' else 4)==0
  if no_match:menu=other[:10]
  rng.shuffle(menu)
  candidates=[{'id':source+':'+i,'label':labels[source][i]} for i in menu]
  row={'id':f'{source}-{source_split}-{source_row}','product_family':family,'source_id':source,'query':query,'candidates':candidates,'relevance':{c['id']:3 if c['id']==source+':'+intent else 0 for c in candidates},'judgment_status':'source-intent-derived-unreviewed','query_kind':'no-match' if no_match else 'intent-paraphrase','augmentation_parent':f'{source}-{source_split}-{source_row}','split_group':family,'source_split':source_split,'source_row':source_row,'source_intent':intent,'intent_family':family,'evaluation_slice':slice_name,'negative_status':'removed-positive-uncertain' if no_match else 'source-class-inequality-unreviewed','license':SOURCES[source]['license'],'source_url':f"https://github.com/{SOURCES[source]['repo']}/tree/{SOURCES[source]['revision']}"}
  partitions[partition].append(row)
 # Real software-setting documentation provides a domain-relevant auxiliary source.
 ui_path=ROOT/'data/ui-settings/pairs.json'
 ui_counts=collections.Counter()
 if ui_path.exists():
  ui=read_json(ui_path)
  ui_source=read_json(ROOT/'data/ui-settings/source.json')
  def ui_split(family):
   bucket=int(digest('vscode-namespace-v1:'+family)[:8],16)%6
   return 'dev' if bucket==0 else 'calibration' if bucket==1 else 'train'
  def tokens(text):return set(re.findall(r'[a-z0-9]+',key(text)))
  def near(a,b):
   # Frozen conservative near-duplicate rule, applied to queries and target labels.
   aa,bb=tokens(a),tokens(b)
   return a==b or (len(aa)>=4 and len(bb)>=4 and len(aa&bb)/max(1,len(aa|bb))>=0.85)
  accepted_ui=[]
  for pair in sorted(ui,key=lambda p:({'train':0,'dev':1,'calibration':2}[ui_split(p['family'])],p['id'])):
   partition=ui_split(pair['family']); normalized=key(pair['query'])
   if normalized in seen:dropped['ui_duplicate_query']+=1;continue
   if any(partition!=other_partition and (near(normalized,key(other['query'])) or near(key(pair['label']),key(other['label']))) for other_partition,other in accepted_ui):
    dropped['ui_cross_split_near_duplicate']+=1;continue
   # Check description-like query overlap with intent-derived queries too. Token
   # overlap threshold makes short common terms insufficient for duplicate status.
   if any(partition!=p and near(normalized,key(row['query'])) for p,rows in partitions.items() for row in rows):
    dropped['ui_external_cross_split_near_duplicate']+=1;continue
   seen[normalized]=partition;accepted_ui.append((partition,pair))
  for partition,pair in accepted_ui:
   pool=[other for p,other in accepted_ui if p==partition and other['id']!=pair['id']]
   rng=random.Random(int(digest('vscode-menu:'+pair['id'])[:16],16));rng.shuffle(pool)
   words=tokens(pair['label']);pool.sort(key=lambda other:-len(words&tokens(other['label'])))
   no_match=int(digest('negative:'+key(pair['query']))[:8],16)%(10 if partition=='train' else 4)==0
   menu=pool[:10] if no_match else [pair]+pool[:9];rng.shuffle(menu)
   candidates=[{'id':'vscode:'+item['id'],'label':item['label']} for item in menu]
   family='vscode:'+pair['family']
   partitions[partition].append({'id':'vscode:'+pair['id'],'product_family':family,'source_id':'vscode-settings','query':pair['query'],'candidates':candidates,'relevance':{c['id']:3 if c['id']=='vscode:'+pair['id'] else 0 for c in candidates},'judgment_status':'documentation-derived-unreviewed','query_kind':'no-match' if no_match else 'setting-description','augmentation_parent':'vscode:'+pair['id'],'split_group':family,'source_split':'source-namespace-partition','source_intent':pair['id'],'intent_family':family,'evaluation_slice':'ui-settings-unseen-namespace' if partition!='train' else 'ui-settings-train','negative_status':'removed-positive-uncertain' if no_match else 'source-association-unreviewed','license':'MIT','source_url':f"{ui_source['source']}/blob/{ui_source['revision']}/{pair['path']}#L{pair['line']}"})
   ui_counts[partition]+=1
 audit={'normalized_query_overlap':{},'held_family_training_candidates':[],'dropped':dict(dropped),'upstream_tests':'CLINC test values never accessed; BANKING test.csv never downloaded; neither evaluated','original_dev_test':'not opened by preparation','human_reviewed_retrieval_rows':0,'seed':'SHA256 wording-v1 plus source-row seeded menu RNG','ui_settings_counts':dict(ui_counts),'ui_namespace_partitions':{pair['family']:partition for partition,pair in accepted_ui} if ui_path.exists() else {},'ui_near_duplicate_rule':'cross-split normalized query or target-label token-set Jaccard >=0.85 with at least 4 tokens, or exact; checked before menus' }
 sets={p:{key(r['query']) for r in rows} for p,rows in partitions.items()}
 for a,b in [('train','dev'),('train','calibration'),('dev','calibration')]:audit['normalized_query_overlap'][a+':'+b]=len(sets[a]&sets[b])
 for row in partitions['train']:
  for candidate in row['candidates']:
   if ':' in candidate['id']:
    source,intent=candidate['id'].split(':',1)
    if source in groups and source+':'+groups[source][intent] in HOLDOUT:audit['held_family_training_candidates'].append(candidate['id'])
 assert not any(audit['normalized_query_overlap'].values())
 assert not audit['held_family_training_candidates']
 counts={}
 for partition,rows in partitions.items():
  payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
  (OUT/f'{partition}.jsonl').write_text(payload)
  counts[partition]={'rows':len(rows),'sources':dict(collections.Counter(r['source_id'] for r in rows)),'slices':dict(collections.Counter(r['evaluation_slice'] for r in rows)),'no_match':sum(not any(r['relevance'].values()) for r in rows),'sha256':hashlib.sha256(payload.encode()).hexdigest()}
 write_json(OUT/'split-audit.json',audit)
 source_provenance={**SOURCES}
 if ui_path.exists():source_provenance['vscode-settings']=ui_source
 write_json(OUT/'manifest.json',{'version':'external-intents-v1','sources':source_provenance,'counts':counts,'holdout_families':HOLDOUT,'intent_groups':groups,'menu_size':10,'banking_per_known_intent_cap':{'train':40,'dev':10,'calibration':10},'judgments':'Source intent class transformed to single relevant label; distractors and removed-positive no-match are not human reviewed. Not a final release test.','domain_limit':'Assistant/banking intent classification is auxiliary supervision, not proof of arbitrary software-interface retrieval.'})
 print(json.dumps(counts,indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');args=parser.parse_args()
 if args.download:download()
 prepare()
