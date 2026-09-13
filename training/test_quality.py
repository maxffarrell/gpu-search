"""Numeric, ranking, abstention and actual-checkpoint regression tests."""
import json
import hashlib
import unittest
from pathlib import Path

import numpy as np
from training.evaluate_expanded import Evaluator, calibrate, rankings, summarize, train_aliases, Tfidf, load_artifact, read_rows, alias_lexical, lexical


class MetricContracts(unittest.TestCase):
    def test_multiple_positives_and_grade_one_count(self):
        row={'query':'q','candidates':[{'id':'a','label':'A'},{'id':'b','label':'B'},{'id':'c','label':'C'}],'relevance':{'a':3,'b':2,'c':0}}
        actual=summarize([row],[[2,1]])
        self.assertAlmostEqual(actual['recall3'],.5)
        self.assertAlmostEqual(actual['mrr'],.5)
        self.assertAlmostEqual(actual['ndcg5'],(3/np.log2(3))/(7+3/np.log2(3)))
        weak={**row,'relevance':{'a':1,'b':0,'c':0}}
        self.assertEqual(summarize([weak],[[0]])['ndcg5'],1)

    def test_hard_tiers_win_against_adversarial_scores(self):
        row={'query':'profile','candidates':[{'id':'x','label':'Profile'},{'id':'l','label':'Profiles'},{'id':'s','label':'Account'}],'relevance':{'x':3,'l':2,'s':0}}
        evaluator=Evaluator([], [row])
        self.assertEqual(rankings([row],[[-1,-1,1]],-1,evaluator.tiers),[[0,1,2]])
        self.assertEqual(rankings([row],[[float('nan'),float('inf'),float('nan')]],0,evaluator.tiers),[[0,1]])

    def test_calibration_controls_ties_and_never_uses_dev(self):
        none={'query':'outside intent','candidates':[{'id':'a','label':'Alpha'}],'relevance':{'a':0}}
        rows=[none]*20;scores=[[.9]]+[[.5]]*19
        cutoff=calibrate(rows,scores)
        self.assertGreater(cutoff,.5)
        self.assertEqual(sum(s[0]>=cutoff for s in scores),1)
        self.assertEqual(calibrate([],[]),1.01)

    def test_zero_weights_cannot_produce_semantics_even_without_cutoff(self):
        row={'query':'coworkers','candidates':[{'id':'a','label':'Members'}],'relevance':{'a':3}}
        evaluator=Evaluator([], [row]);params={key:np.zeros((1024,16),np.float32) for key in ['word','char']}
        scores=evaluator.scores(params)
        self.assertEqual(rankings([row],scores,-1,evaluator.tiers),[[]])
        self.assertEqual(evaluator.evaluate(params)['semanticNdcg5'],0)
        none={**row,'relevance':{'a':0}}
        self.assertEqual(summarize([none],[[]])['noMatchSemanticRate'],0)

    def test_alias_and_tfidf_fit_are_train_only(self):
        train=[{'query':'my coworkers','candidates':[{'id':'a','label':'Members'},{'id':'b','label':'Billing'}],'relevance':{'a':3,'b':0}}]
        self.assertEqual(train_aliases(train),{'Members':['my coworkers']})
        tf=Tfidf(train)
        self.assertEqual(tf.vector('ZZZZZZ'),{})
        self.assertGreater(tf.score([{'query':'my coworkers','candidates':[{'id':'a','label':'Members'}]}],True)[0][0],.999)

    def test_optimized_full_alias_matcher_retains_lexical_contract(self):
        candidate={'id':'p','label':'Personal Settings','aliases':['myProfile']}
        aliases={'Personal Settings':['my information','Profile','Application Programming Interface','fooBar']}
        for query in ['Personal Settings','personalSettings','Profile','profiles','profle','my information','my info','api','x','foo-bar','xqzv']:
            self.assertEqual(alias_lexical(query,candidate,aliases),lexical(query,candidate,aliases))


class SelectedArtifactQuality(unittest.TestCase):
    def test_selected_trained_artifact_meets_measured_quality_floors(self):
        floors=json.loads(Path('eval/expanded-quality-floors.json').read_text())
        for path,expected in floors['datasetSha256'].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),expected,'Frozen regression corpus changed')
        params,manifest=load_artifact(floors['artifact'])
        evaluator=Evaluator(read_rows('data/expanded/calibration.jsonl'),read_rows('data/expanded/dev.jsonl'))
        metrics=evaluator.evaluate(params,manifest['featureFamily'])
        self.assertGreaterEqual(metrics['semanticNdcg5'],floors['minimumSemanticNdcg5'])
        self.assertGreaterEqual(metrics['semanticCoverage'],floors['minimumSemanticCoverage'])
        self.assertLessEqual(metrics['noMatchSemanticRate'],floors['maximumNoMatchSemanticRate'])
        # Same assertion must reject a degraded model, rather than merely matching a snapshot.
        degraded={key:np.zeros_like(value) for key,value in params.items()}
        self.assertLess(evaluator.evaluate(degraded,manifest['featureFamily'])['semanticNdcg5'],floors['minimumSemanticNdcg5'])
        transfer=Evaluator(read_rows('data/expanded/calibration.jsonl'),read_rows('data/dev.jsonl'))
        details=transfer.evaluate(params,manifest['featureFamily'],detail=True)
        raw=rankings(transfer.rows,details['scores'],-1,transfer.tiers,raw=True)
        self.assertGreaterEqual(summarize(transfer.rows,raw)['ndcg5'],floors['minimumTransferRawOverallNdcg5'])
        self.assertGreaterEqual(details['overallNdcg5'],floors['minimumTransferCombinedOverallNdcg5'])


if __name__=='__main__':unittest.main()
