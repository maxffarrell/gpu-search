"""Extract author-provided desktop search metadata, never invent negative labels.

python -m training.prepare_navigation --download
python -m training.prepare_navigation
Only original English fields are read. No source code is executed.
"""
import argparse, collections, hashlib, io, json, pathlib, re, tarfile, urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'data/navigation'
SOURCES=[
 {'provider':'gnome','product':'gnome-control-center','repo':'GNOME/gnome-control-center','revision':'cd79a897190989ca395c6f00962f0929734103c7'},
 {'provider':'kde','product':'plasma-desktop','repo':'KDE/plasma-desktop','revision':'19dae94ec1d563d095b08602dd5eed60d485a5e0'},
 {'provider':'kde','product':'plasma-workspace','repo':'KDE/plasma-workspace','revision':'df3bba49a2c0657d4165f6511b1771138ccc75ea'},
 {'provider':'xfce','product':'xfce4-settings','repo':'xfce-mirror/xfce4-settings','revision':'488c919d23c979c9b28abd192c05b9b2310fa732'},
]

def write(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def wanted(path):
 p=pathlib.PurePosixPath(path);name=p.name
 license_file=(name.upper().startswith(('COPYING','LICENSE','AUTHORS','COPYRIGHT')) or name in ['REUSE.toml','dep5'] or 'LICENSES' in p.parts or name.endswith('.license'))
 metadata=('.desktop' in name or (name.endswith('.json') and ('kcms' in p.parts or 'kcm' in name)))
 return license_file or metadata

def download():
 manifests=[]
 for source in SOURCES:
  url=f"https://codeload.github.com/{source['repo']}/tar.gz/{source['revision']}"
  blob=urllib.request.urlopen(url,timeout=120).read();files=[];base=OUT/'raw'/source['product']
  with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:
   for entry in archive:
    if not entry.isfile():continue
    parts=pathlib.PurePosixPath(entry.name).parts[1:]
    if not parts or any(p in ['..','.'] for p in parts):continue
    relative='/'.join(parts)
    if not wanted(relative):continue
    data=archive.extractfile(entry).read();target=base.joinpath(*parts);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    files.append({'path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
  manifests.append({**source,'archiveUrl':url,'archiveSha256':hashlib.sha256(blob).hexdigest(),'archiveBytes':len(blob),'retainedFiles':files})
  print(source['product'],'retained',len(files),'metadata/license files',flush=True)
 write(OUT/'downloads.json',manifests)

def english_desktop(text):
 fields={};line_numbers={};section=None
 for line_no,line in enumerate(text.splitlines(),1):
  stripped=line.strip()
  if stripped.startswith('[') and stripped.endswith(']'):section=stripped[1:-1];continue
  if section!='Desktop Entry' or not stripped or stripped.startswith('#') or '=' not in stripped:continue
  name,value=stripped.split('=',1)
  if name in ['Name','_Name','Comment','_Comment','Keywords','_Keywords','X-KDE-Keywords','GenericName','_GenericName','X-KDE-System-Settings-Parent-Category','X-KDE-ServiceTypes','Type','NoDisplay','Hidden','OnlyShowIn','Categories']:
   name=name.lstrip('_');fields[name]=value;line_numbers[name]=line_no
 return fields,line_numbers

def unescape(text):
 # Desktop Entry string escapes; list separators are parsed beforehand.
 return re.sub(r'\\([sntr\\;])',lambda m:{'s':' ','n':'\n','t':'\t','r':'\r','\\':'\\',';':';'}[m[1]],text)
def split_keywords(value,sep):
 if isinstance(value,list):return [str(x).strip() for x in value if isinstance(x,str) and x.strip()]
 return [unescape(s.strip()) for s in re.split(r'(?<!\\)'+re.escape(sep),value) if s.strip()]

def licensing(source,path,text):
 relative=path.relative_to(OUT/'raw'/source['product']).as_posix()
 expressions=re.findall(r'SPDX-License-Identifier:\s*([^\n"<>]+)',text)
 sidecar=pathlib.Path(str(path)+'.license')
 if sidecar.exists():expressions+=re.findall(r'SPDX-License-Identifier:\s*([^\n]+)',sidecar.read_text())
 declarations=[]
 base=OUT/'raw'/source['product']
 for filename in ['COPYING','COPYING.LIB','COPYING.txt','LICENSE','LICENSE.txt','REUSE.toml','.reuse/dep5']:
  f=base/filename
  if f.exists():declarations.append(str(f.relative_to(OUT)))
 for f in (base/'LICENSES').glob('*') if (base/'LICENSES').exists() else []:declarations.append(str(f.relative_to(OUT)))
 return {'fileSpdx':list(dict.fromkeys(s.strip().rstrip('*/ ') for s in expressions)), 'evidenceFiles':sorted(declarations), 'sourceLicenseNotice':'GPL version 2 text at repository root COPYING' if source['provider'] in ['gnome','xfce'] else 'mixed per-file licenses; retained upstream LICENSES collection does not assign one license to this metadata file','status':'file-SPDX' if expressions else 'upstream-license-evidence-retained; no file-SPDX found; do not assume MIT'}

def extract():
 panels=[];skip=collections.Counter()
 for source in SOURCES:
  base=OUT/'raw'/source['product']
  if not base.exists():raise FileNotFoundError(f'Run --download first: {base}')
  for path in sorted(base.rglob('*')):
   if not path.is_file():continue
   relative=path.relative_to(base).as_posix()
   if '.desktop' not in path.name and not (path.suffix=='.json' and ('kcms' in path.parts or 'kcm' in path.name)):continue
   text=path.read_text(errors='strict');fields={};lines={};metadata_type='desktop-entry'
   if path.suffix=='.json':
    try:obj=json.loads(text)
    except json.JSONDecodeError:skip['nonliteral-json']+=1;continue
    plugin=obj.get('KPlugin',{})
    if not isinstance(plugin,dict):continue
    fields={'Name':plugin.get('Name',''),'Comment':plugin.get('Description',''),'Keywords':obj.get('X-KDE-Keywords',plugin.get('X-KDE-Keywords',''))}
    if not fields['Keywords']:skip['json-without-authored-keywords']+=1;continue
    sep=',';metadata_type='kcm-json'
    for key,lookup in [('Name','Name'),('Comment','Description'),('Keywords','X-KDE-Keywords')]:
     lines[key]=next((i for i,l in enumerate(text.splitlines(),1) if re.match(r'\s*"'+re.escape(lookup)+r'"\s*:',l)),1)
   else:
    fields,lines=english_desktop(text)
    if source['provider']=='gnome' and not relative.startswith('panels/'):skip['gnome-nonpanel']+=1;continue
    if source['provider']=='kde' and 'X-KDE-Keywords' not in fields:skip['kde-nonkcm']+=1;continue
    sep=',' if 'X-KDE-Keywords' in fields else ';'
    if 'X-KDE-Keywords' in fields:fields['Keywords']=fields['X-KDE-Keywords'];lines['Keywords']=lines['X-KDE-Keywords']
   if not fields.get('Name') or not fields.get('Keywords'):skip['missing-name-or-keywords']+=1;continue
   keywords=split_keywords(fields['Keywords'],sep)
   if not keywords:continue
   label=unescape(fields['Name']);description=unescape(fields.get('Comment',''))
   role='settings-panel' if source['provider']=='gnome' else 'settings-module' if source['provider']=='kde' else 'settings-panel'
   if source['provider']=='xfce':
    if relative.startswith('xfsettingsd/'):role='background-service'
    elif relative.startswith('xfce4-settings-manager/'):role='settings-catalog'
    elif '/mime-settings/' in relative and path.name!='xfce4-mime-settings.desktop.in':role='preferred-application-launcher'
   if '@' in label and ('@'==label[0] or label.endswith('@')):skip['unresolved-name-template']+=1;continue
   panel_id=f"{source['provider']}:{source['product']}:{relative}"
   source_url=f"https://github.com/{source['repo']}/blob/{source['revision']}/{relative}"
   panels.append({'id':panel_id,'provider':source['provider'],'product':source['product'],'lineage_id':source['provider']+':'+source['product'],'label':label,'description':description,'keywords':keywords,'metadata_type':metadata_type,'entry_role':role,'desktop_flags':{k:fields[k] for k in ['Type','NoDisplay','Hidden','OnlyShowIn','Categories'] if k in fields},'source_path':relative,'source_url':source_url,'revision':source['revision'],'field_lines':lines,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'license':licensing(source,path,text),'judgment_status':'upstream-author-search-keyword; not independently human-reviewed retrieval'})
 pairs=[];shared=collections.defaultdict(list)
 for panel in panels:
  seen=set()
  for keyword in panel['keywords']:
   if keyword in seen:continue
   seen.add(keyword)
   pair={'id':panel['id']+':keyword:'+hashlib.sha256(keyword.encode()).hexdigest()[:16],'provider':panel['provider'],'product':panel['product'],'lineage_id':panel['lineage_id'],'query':keyword,'destination_id':panel['id'],'label':panel['label'],'source_url':panel['source_url']+'#L'+str(panel['field_lines'].get('Keywords',1)),'source_field':'X-KDE-Keywords' if panel['provider']=='kde' else 'Keywords','positive_status':'inherited-author-keyword','entry_role':panel['entry_role'],'negative_status':'unjudged; absence from metadata is not evidence of irrelevance','license':panel['license']}
   pairs.append(pair);shared[(panel['lineage_id'],keyword.casefold().strip())].append(panel['id'])
 queries=[{'lineage_id':lineage,'query_normalized_group':query,'positive_destination_ids':list(dict.fromkeys(ids)),'other_destinations':'unjudged'} for (lineage,query),ids in sorted(shared.items())]
 write(OUT/'panels.json',panels);write(OUT/'keyword-pairs.json',pairs);write(OUT/'query-groups.json',queries)
 counts={'panels':len(panels),'keyword_pairs':len(pairs),'query_groups':len(queries),'entry_roles':dict(collections.Counter(p['entry_role'] for p in panels)),'multiple_positive_query_groups':sum(len(q['positive_destination_ids'])>1 for q in queries),'providers':{},'skipped':dict(skip),'split_policy':'not assigned; root must freeze lineage groups before training/evaluation','relevance_policy':'author-listed positive destinations only; all unlisted destinations unjudged','sources':SOURCES}
 for provider in sorted({p['provider'] for p in panels}):counts['providers'][provider]={'panels':sum(p['provider']==provider for p in panels),'keyword_pairs':sum(p['provider']==provider for p in pairs)}
 write(OUT/'manifest.json',counts);print(json.dumps(counts,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');args=p.parse_args()
 if args.download:download()
 extract()
