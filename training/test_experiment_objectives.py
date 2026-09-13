"""Portable checks for objective behavior and saved experimental gate evidence."""
import json
import math
import unittest
from pathlib import Path
import numpy as np
from training.objective_reference import objective
from training.export import quantize


class ObjectiveTests(unittest.TestCase):
    def test_single_positive_is_negative_log_probability(self):
        actual=objective([[0.,0.]],[ [True,False] ],[[True,True]],temperature=1.)
        self.assertAlmostEqual(actual,math.log(2))

    def test_multiple_positives_share_probability_mass(self):
        actual=objective([[0.,0.,0.]],[[True,True,False]],[[True,True,True]],temperature=1.)
        self.assertAlmostEqual(actual,math.log(3/2))
        self.assertEqual(objective([[.4,-.8]],[[True,True]],[[True,True]]),0.)

    def test_unjudged_and_padding_cannot_change_objective(self):
        expected=objective([[.2,-.1]],[[True,False]],[[True,True]],hard_margin_weight=.25)
        actual=objective([[.2,-.1,1e9,float('nan')]],[[True,False,False,False]],[[True,True,False,False]],hard_margin_weight=.25)
        self.assertEqual(actual,expected)

    def test_direction_of_positive_and_negative_evidence(self):
        masks=([[True,False]],[[True,True]])
        base=objective([[.2,.1]],*masks)
        self.assertLess(objective([[.5,.1]],*masks),base)
        self.assertGreater(objective([[.2,.4]],*masks),base)

    def test_no_match_has_no_empty_positive_numerator(self):
        masks=([[False,False]],[[True,True]])
        self.assertEqual(objective([[.6,.1]],*masks),0.)
        self.assertAlmostEqual(objective([[.6,.1]],*masks,no_match_weight=.25,no_match_target=.4),.01)
        self.assertEqual(objective([[.3,.1]],*masks,no_match_weight=.25,no_match_target=.4),0.)

    def test_mixed_batch_uses_separate_positive_and_no_match_means(self):
        positive=objective([[.2,.1]],[[True,False]],[[True,True]])
        actual=objective([[.2,.1],[.6,.1]],[[True,False],[False,False]],[[True,True],[True,True]],no_match_weight=.25,no_match_target=.4)
        self.assertAlmostEqual(actual,positive+.01)

    def test_hard_margin_uses_hardest_judged_negative(self):
        kwargs={'temperature':1.,'hard_margin':.15}
        base=objective([[.3,.6,-1]],[[True,False,False]],[[True,True,True]],**kwargs)
        augmented=objective([[.3,.6,-1]],[[True,False,False]],[[True,True,True]],hard_margin_weight=.25,**kwargs)
        self.assertAlmostEqual(augmented-base,.25*.45)

    def test_invalid_masks_and_nonfinite_judgments_reject(self):
        with self.assertRaises(ValueError):objective([[0.]],[[True]],[[False]])
        with self.assertRaises(ValueError):objective([[0.]],[[False]],[[False]])
        with self.assertRaises(ValueError):objective([[float('nan')]],[[True]],[[True]])
        with self.assertRaises(ValueError):objective([[0.]],[[True]],[[True]],temperature=0)

    def test_nonfinite_weights_never_export(self):
        for value in [float('nan'),float('inf'),-float('inf')]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):quantize(np.array([0.,value]),8)


class ReportAuditTests(unittest.TestCase):
    def test_capacity_flags_follow_actual_metrics(self):
        report=json.loads(Path('eval/training-experiments-capacity.json').read_text())
        self.assertEqual(len(report['runs']),3)
        for run,quant in zip(report['runs'],report['quantization']):
            self.assertEqual(run['passesDevNoMatchGate'],run['devNoMatchSemanticRate']<=.05)
            for kind,bits in [('int8',8),('int6',6)]:
                result=quant[kind]
                self.assertEqual(result['passesDevNoMatchGate'],result['overall']['noMatchSemanticRate']<=.05)
                self.assertEqual(result['passesQuantizationLossGate'],result['semanticNdcgLoss']<=.01)
                self.assertEqual(result['payloadBytes'],2048*24*bits//8)
                manifest=json.loads((Path(result['artifact'])/'manifest.json').read_text())
                self.assertIsNone(manifest['validatedSemanticCutoff'])

    def test_infeasible_checkpoints_are_explicitly_unapproved(self):
        report=json.loads(Path('eval/training-experiments.json').read_text())
        self.assertEqual(len(report['runs']),21)
        for run in report['runs']:
            meta=json.loads((Path(run['checkpoint'])/'training.json').read_text())
            self.assertEqual(meta['selectedPassesDevNoMatchGate'],run['passesDevNoMatchGate'])
            self.assertEqual(run['passesDevNoMatchGate'],run['devNoMatchSemanticRate']<=.05)
            self.assertEqual(meta['int8WeightBytes'],32768)
            if not run['passesDevNoMatchGate']:self.assertTrue(meta['status'].startswith('NOT SELECTED:'))


if __name__=='__main__':unittest.main()
