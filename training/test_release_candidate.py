import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from training import evaluate_release_candidate as release
from training import features_bigram


class ReleaseCandidateContracts(unittest.TestCase):
    def test_research_candidate_preserves_development_gain_at_same_weight_size(self):
        # Development regression only: this test does not read the consumed holdout.
        directory='packages/model/candidate-v2'
        metadata=release.asset_metadata(directory)
        self.assertEqual(metadata['payloadBytes'],32768)
        rows=release.read_rows('data/expanded/dev.jsonl');cal=release.read_rows('data/expanded/calibration.jsonl')
        cutoff=release.calibrate(cal,release.model_scores(directory,cal))
        evaluator=release.Evaluator([],rows)
        ranked=release.rankings(rows,release.model_scores(directory,rows),cutoff,evaluator.tiers)
        actual=release.describe(evaluator,ranked)
        self.assertGreaterEqual(actual['semanticOnly']['ndcg5'],.51)
        self.assertGreaterEqual(actual['semanticOnly']['coverage'],.54)
        self.assertLessEqual(actual['overall']['noMatchSemanticRate'],.05)

    def test_bigram_dispatch_matches_portable_reference_and_not_unigram(self):
        rng=np.random.default_rng(17)
        params={key:rng.normal(size=(1024,16)).astype(np.float32) for key in ['word','char']}
        manifest={'featureVersion':features_bigram.VERSION,'architecture':'pooled','featureFamily':'both'}
        rows=[{'query':'please show my account','relevance':{'a':3},'candidates':[{'id':'a','label':'my account details','aliases':['profile information'],'context':'personal settings'}]}]
        with patch.object(release,'load_artifact',return_value=(params,manifest)):
            got=release.model_scores('unused',rows)[0][0]
        expected=float(features_bigram.encode_numpy(params,[rows[0]['query']])[0]@features_bigram.candidate_vectors(params,rows[0]['candidates'])[0])
        self.assertAlmostEqual(got,expected,places=6)
        baseline={**manifest,'featureVersion':'gpu-search-features-v1'}
        with patch.object(release,'load_artifact',return_value=(params,baseline)):
            unigram=release.model_scores('unused',rows)[0][0]
        self.assertGreater(abs(got-unigram),1e-5)

    def test_unknown_features_fail_closed(self):
        with patch.object(release,'load_artifact',return_value=({}, {'featureVersion':'unknown'})):
            with self.assertRaisesRegex(ValueError,'explicit --adapter'):release.model_scores('unused',[])

    def test_sealed_claim_requires_frozen_matching_hash_and_is_one_time(self):
        with tempfile.TemporaryDirectory() as temp:
            selection=Path(temp)/'selection.json';receipt=Path(temp)/'receipt.json'
            metadata={'manifestSha256':'m','payloadSha256':'p'}
            selection.write_text(json.dumps({'selectionFrozen':True,'manifestSha256':'m','payloadSha256':'wrong'}))
            with self.assertRaisesRegex(ValueError,'mismatch'):release.claim_sealed_access(selection,metadata,receipt)
            self.assertFalse(receipt.exists())
            selection.write_text(json.dumps({'selectionFrozen':True,**metadata}))
            release.claim_sealed_access(selection,metadata,receipt)
            with self.assertRaises(FileExistsError):release.claim_sealed_access(selection,metadata,receipt)

    def test_default_evaluation_never_opens_or_hashes_sealed_input(self):
        seen=[];reader=release.read_rows;hasher=release.digest
        def guarded_read(path):
            self.assertNotEqual(str(path),'data/test.jsonl');seen.append(str(path));return reader(path)
        def guarded_hash(path):
            self.assertNotEqual(str(path),'data/test.jsonl');return hasher(path)
        with patch.object(release,'read_rows',side_effect=guarded_read),patch.object(release,'digest',side_effect=guarded_hash):
            report=release.evaluate()
        self.assertFalse(report['sealedExecuted']);self.assertIsNone(report['sealedAccess'])
        self.assertEqual(set(seen),{'data/expanded/calibration.jsonl','data/expanded/dev.jsonl','data/dev.jsonl'})
        self.assertTrue(report['allRegressionFloorsPassed'])
        self.assertEqual(report['partitions']['expanded-dev']['pairedCandidateMinusBaseline']['calibrated']['delta'],0)


if __name__=='__main__':unittest.main()
