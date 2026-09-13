"""Train-seeded WordNet synonym edges, positive supervision only."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET

from training.features import normalize, tokens

SOURCE='https://github.com/globalwordnet/english-wordnet/releases/download/2025-edition/english-wordnet-2025.xml.gz'
SHA256='9ca6d1dcb75f822fdd66617f7d9da48142ace38dd544d6ad5e2feca1674ad3fe'
REVISION='dc343f2683279ecbb13fab4e2fd778d7b162d287'


def prepare(source,output):
    path=Path(source)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SHA256:raise ValueError('Pinned WordNet source checksum mismatch')
    train_paths=[Path('data/expanded/train.jsonl'),Path('data/navigation-splits/train.jsonl')]
    seed_words=set()
    for train_path in train_paths:
        for line in train_path.read_text().splitlines():
            row=json.loads(line)
            for candidate in row['candidates']:
                if row['relevance'].get(candidate['id'],0)<2:continue
                seed_words.update(word for word in tokens(candidate['label']) if re.fullmatch('[a-z]{3,32}',word))
    synsets=defaultdict(set);senses=defaultdict(set)
    with gzip.open(path,'rb') as stream:
        for _,element in ET.iterparse(stream,events=['end']):
            if element.tag!='LexicalEntry':continue
            lemma=element.find('Lemma')
            if lemma is not None and lemma.attrib.get('partOfSpeech') in ['n','v','a','s']:
                term=normalize(lemma.attrib['writtenForm'])
                if re.fullmatch('[a-z][a-z -]{2,63}',term) and len(tokens(term))<=4:
                    for sense in element.findall('Sense'):
                        identifier=sense.attrib['synset'];synsets[identifier].add(term);senses[term].add(identifier)
            element.clear()
    pairs=defaultdict(set)
    for synset,members in sorted(synsets.items()):
        anchors=members & seed_words
        if not anchors:continue
        for anchor in anchors:
            for other in members:
                if anchor==other or max(len(senses[anchor]),len(senses[other]))>4:continue
                pairs[tuple(sorted([anchor,other]))].add(synset)
    records=[{'left':left,'right':right,'synsets':sorted(ids),'judgment':'same published synset; sense-specific, not unconditional interface equivalence'} for (left,right),ids in sorted(pairs.items())]
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    payload=json.dumps(records,ensure_ascii=False,indent=2)+'\n';(out/'positive-pairs.json').write_text(payload)
    manifest={'source':SOURCE,'sourceSha256':SHA256,'revision':REVISION,'license':'Open English WordNet CC BY4.0 with Princeton WordNet license attribution',
              'seedSources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in train_paths},'seedWords':len(seed_words),'pairs':len(records),
              'pairSha256':hashlib.sha256(payload.encode()).hexdigest(),'filter':'Only noun/verb/adjective synsets directly touching TRAIN positive-label words; ASCII terms3-64chars<=4tokens, <=4senses per term.',
              'negativeJudgments':'none','holdoutAccess':'No navigation dev/holdout, expanded dev/calibration or consumed synthetic test is read.',
              'use':'Optional positive-only regularization, no shipped dictionary/lookup; polysemy still requires caution.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='/tmp/gpu-search-wordnet-2025.xml.gz');parser.add_argument('--output',default='data/lexicon');parser.add_argument('--download',action='store_true');args=parser.parse_args()
    if args.download:Path(args.source).write_bytes(urllib.request.urlopen(SOURCE).read())
    prepare(args.source,args.output)
