import unittest,tempfile,json
from pathlib import Path
from unittest import mock
from training.prepare_typo import corruptions
from training.evaluate_typo import claim

class TypoTests(unittest.TestCase):
    def test_real_typo_operators(self):
        variants=corruptions('profile')
        for text in ['profle','prfoile','profiile']:self.assertIn(text,variants)
        self.assertNotIn('profile',variants)
        self.assertEqual(variants,corruptions('profile'))
    def test_short_tokens_and_negation_preserved(self):
        self.assertEqual(corruptions('not api'),[])
        self.assertTrue(all(t.startswith('not ') for t in corruptions('not enabled')))
    def test_label_splits_and_pair_sources(self):
        m=json.loads(Path('data/typo/manifest.json').read_text());sets=[set(p['cleanLabels']) for p in m['partitions'].values()]
        for i,left in enumerate(sets):
            for right in sets[i+1:]:self.assertFalse(left&right)
        pairs=json.loads(Path('data/typo/train-pairs.json').read_text())
        self.assertTrue(all(clean in sets[0] for _,clean in pairs))
        self.assertNotIn('profile',sets[0])
    def test_exact_freeze_and_exclusive_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection=Path(tmp)/'selection.json';receipt=Path(tmp)/'receipt.json';meta={'payloadBytes':32768,'payloadSha256':'a','manifestSha256':'b'}
            with mock.patch('training.evaluate_typo.asset_metadata',return_value=meta):
                selection.write_text(json.dumps({**meta,'selectionFrozen':False}))
                with self.assertRaises(ValueError):claim(selection,'unused',receipt)
                self.assertFalse(receipt.exists())
                selection.write_text(json.dumps({**meta,'selectionFrozen':True}));claim(selection,'unused',receipt)
                with self.assertRaises(FileExistsError):claim(selection,'unused',receipt)
if __name__=='__main__':unittest.main()
