"""Portable evaluator tests with synthetic rows and isolated freeze-guard files.

Never loads navigation datasets, models, or the real holdout access receipt.
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training import evaluate_navigation as navigation


def row(positive='a',grades=None):
    return {'query':'synthetic query','candidates':[{'id':'a','label':'Alpha'},{'id':'b','label':'Beta'},{'id':'unknown','label':'Unjudged'}],'relevance':grades if grades is not None else {positive:3}}


class NavigationMetricsTests(unittest.TestCase):
    def test_partial_unknowns_are_not_a_false_positive_estimate(self):
        result=navigation.metrics([row()],[[2]],[0])
        self.assertEqual(result['ndcg5'],0.)
        self.assertEqual(result['success1'],0.)
        self.assertEqual(result['coverage'],1.)
        self.assertIsNone(result['noMatchRate'])
        self.assertIn('unjudged',result['judgments'])

    def test_multiple_positive_success_is_not_recall_fraction(self):
        result=navigation.metrics([row(grades={'a':3,'b':2})],[[1,0]],[0])
        self.assertEqual(result['success1'],1.)
        expected=(3+7/math.log2(3))/(7+3/math.log2(3))
        self.assertAlmostEqual(result['ndcg5'],expected)
        self.assertEqual(result['success3'],1.)

    def test_empty_return_is_abstention_not_measured_no_match(self):
        result=navigation.metrics([row()],[[]],[0])
        self.assertEqual(result['coverage'],0.)
        self.assertEqual(result['ndcg5'],0.)
        self.assertEqual(result['success5'],0.)
        self.assertIsNone(result['noMatchRate'])
        self.assertEqual(navigation.metrics([row()],[[]],[]),{'queries':0})

    def test_requested_slice_only(self):
        result=navigation.metrics([row(),row('b')],[[],[1]],[1])
        self.assertEqual(result['queries'],1)
        self.assertEqual(result['ndcg5'],1.)
        self.assertEqual(result['coverage'],1.)

    def test_bootstrap_groups_repeated_destinations_and_preserves_pairing(self):
        rows=[row(),row(),row('b')]
        left=[[0],[0],[1]];right=[[],[],[]]
        result=navigation.paired(rows,left,right,[0,1,2])
        self.assertEqual(result['count'],3)
        self.assertEqual(result['documentedDestinationGroups'],2)
        self.assertEqual(result['delta'],1.)
        self.assertEqual(result['queryBootstrap95'],[1.,1.])
        self.assertEqual(result['destinationGroupBootstrap95'],[1.,1.])
        reverse=navigation.paired(rows,right,left,[0,1,2])
        self.assertEqual(reverse['delta'],-1.)
        self.assertEqual(navigation.paired(rows,left,right,[]),{'count':0})


class FrozenSelectionGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='navigation-freeze-test-')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/'eval').mkdir()
        self.metadata={'payloadBytes':32768,'payloadSha256':'synthetic-payload-hash','manifestSha256':'synthetic-manifest-hash','modelId':'synthetic-model'}
        self.receipt=self.root/'eval/navigation-holdout-access.json'
        self.selection=self.root/'selection.json'

    def write_selection(self,**updates):
        data={'selectionFrozen':True,'payloadSha256':self.metadata['payloadSha256'],'manifestSha256':self.metadata['manifestSha256']}
        data.update(updates);self.selection.write_text(json.dumps(data))

    def invoke(self,selection=True,metadata=None):
        argv=['evaluate_navigation','--candidate','synthetic-artifact','--partition','holdout']
        if selection:argv+=['--selection-record',str(self.selection)]
        real_path=Path
        def isolated_path(value):
            candidate=real_path(value)
            return candidate if candidate.is_absolute() else self.root/candidate
        # A sentinel stops immediately after a successful preflight; no dataset is read.
        with mock.patch.object(sys,'argv',argv), mock.patch.object(navigation,'Path',side_effect=isolated_path), mock.patch.object(navigation,'asset_metadata',return_value=metadata or self.metadata), mock.patch.object(navigation,'read_rows',side_effect=RuntimeError('SYNTHETIC_READ_SENTINEL')) as reader:
            try:navigation.main()
            finally:self.read_calls=reader.call_count

    def test_requires_frozen_record_before_any_data_access(self):
        with self.assertRaisesRegex(ValueError,'frozen selection record'):self.invoke(selection=False)
        self.assertEqual(self.read_calls,0)
        self.assertFalse(self.receipt.exists())

    def test_requires_both_exact_hashes_and_frozen_flag(self):
        for change in [{'selectionFrozen':False},{'payloadSha256':'changed'},{'manifestSha256':'changed'}]:
            with self.subTest(change=change):
                self.write_selection(**change)
                with self.assertRaisesRegex(ValueError,'exact candidate'):self.invoke()
                self.assertEqual(self.read_calls,0)
                self.assertFalse(self.receipt.exists())

    def test_payload_budget_rejects_before_receipt_or_data_access(self):
        self.write_selection()
        with self.assertRaisesRegex(ValueError,'raw model budget'):self.invoke(metadata={**self.metadata,'payloadBytes':32769})
        self.assertEqual(self.read_calls,0)
        self.assertFalse(self.receipt.exists())

    def test_matching_freeze_creates_receipt_before_mocked_read(self):
        self.write_selection()
        with self.assertRaisesRegex(RuntimeError,'SYNTHETIC_READ_SENTINEL'):self.invoke()
        self.assertEqual(self.read_calls,1)
        receipt=json.loads(self.receipt.read_text())
        self.assertEqual(receipt['candidate'],self.metadata)
        self.assertEqual(receipt['selectionSha256'],navigation.digest(self.selection))
        self.assertEqual(receipt['selection'],str(self.selection))

    def test_receipt_is_exclusive_and_cannot_be_overwritten(self):
        self.write_selection();self.receipt.write_text('original synthetic receipt')
        with self.assertRaises(FileExistsError):self.invoke()
        self.assertEqual(self.read_calls,0)
        self.assertEqual(self.receipt.read_text(),'original synthetic receipt')


if __name__=='__main__':unittest.main()
