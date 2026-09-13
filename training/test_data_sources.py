"""Source/split contracts that prevent accidental evaluation contamination."""
import json
import unittest
from pathlib import Path
from training.features import normalize

ROOT=Path('data/expanded')
def read(name):
    return [json.loads(line) for line in (ROOT/f'{name}.jsonl').read_text().splitlines()]

class DataContracts(unittest.TestCase):
    def test_expanded_partitions_have_no_duplicate_queries(self):
        partitions={name:read(name) for name in ['train','dev','calibration']}
        query_sets={name:{normalize(r['query']) for r in rows} for name,rows in partitions.items()}
        for a,b in [('train','dev'),('train','calibration'),('dev','calibration')]:
            self.assertFalse(query_sets[a]&query_sets[b],f'{a}/{b} normalized query leakage')
        self.assertGreater(len(partitions['train']),1000)
        self.assertGreater(len(partitions['dev']),100)
        self.assertGreater(len(partitions['calibration']),100)

    def test_records_fit_the_runtime_and_have_explicit_judgments(self):
        for split in ['train','dev','calibration']:
            rows=read(split)
            self.assertEqual(len(rows),len({r['id'] for r in rows}))
            for r in rows:
                self.assertTrue(0<len(r['query'])<=256,r['id'])
                self.assertTrue(r['source_id'],r['id'])
                self.assertTrue(r['judgment_status'],r['id'])
                ids=[c['id'] for c in r['candidates']]
                self.assertEqual(len(ids),len(set(ids)),r['id'])
                self.assertEqual(set(r['relevance']),set(ids),r['id'])
                for c in r['candidates']:
                    self.assertTrue(0<len(c['label'])<=256,r['id'])
                    self.assertLessEqual(len(c.get('aliases',[])),8,r['id'])
                if r['query_kind']=='no-match':self.assertTrue(all(v==0 for v in r['relevance'].values()),r['id'])
                else:self.assertTrue(any(v>=2 for v in r['relevance'].values()),r['id'])

    def test_original_development_records_are_not_training_records(self):
        original={r['id'] for r in map(json.loads,Path('data/dev.jsonl').read_text().splitlines())}
        self.assertFalse(original&{r['id'] for r in read('train')})
        self.assertTrue(Path('data/ui-settings/LICENSE-vscode.txt').is_file())

if __name__=='__main__':unittest.main()
