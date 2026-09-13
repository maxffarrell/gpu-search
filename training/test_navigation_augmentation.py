import unittest
from training.navigation_augmentation import shorten_query,augment_positive_rows,documented_positive_pairs

class NavigationPriorTests(unittest.TestCase):
    def test_preserve_negation_and_action(self):
        self.assertEqual(shorten_query('Please do not delete my account'),'do not delete my account')
        self.assertEqual(shorten_query('can you disable password autofill'),'disable password autofill')
        self.assertEqual(shorten_query('how do i not enable notifications'),'not enable notifications')
    def test_bounded_prefix_only(self):
        self.assertIsNone(shorten_query('can you reset'))
        self.assertIsNone(shorten_query('display can you help'))
        self.assertIsNone(shorten_query('please, change settings'))
        self.assertEqual(shorten_query('PLEASE tell me keyboard settings'),'tell me keyboard settings')
    def test_only_positive_anchor_rows_are_augmented(self):
        rows=[{'id':'yes','query':'please enable notifications','candidates':[{'id':'a','label':'Notifications'}],'relevance':{'a':3}}, {'id':'no','query':'please delete records','candidates':[{'id':'b','label':'Records'}],'relevance':{'b':0}}]
        out=augment_positive_rows(rows)
        self.assertEqual(len(out),1)
        self.assertEqual(out[0]['query'],'enable notifications')
        self.assertEqual(out[0]['relevance'],{'a':3})
        self.assertEqual(rows[0]['query'],'please enable notifications')
    def test_all_documented_positives_without_unknown_negatives(self):
        rows=[{'query':'access','candidates':[{'id':'a','label':'Users'},{'id':'b','label':'Security'},{'id':'c','label':'Other'}],'relevance':{'a':3,'b':2}}]
        self.assertEqual(documented_positive_pairs(rows),[('access','Users'),('access','Security')])
        self.assertNotIn('c',rows[0]['relevance'])
    def test_absent_positive_candidate_is_invalid(self):
        with self.assertRaises(ValueError):documented_positive_pairs([{'query':'test','candidates':[],'relevance':{'missing':3}}])

if __name__=='__main__':unittest.main()
