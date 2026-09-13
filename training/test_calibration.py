import unittest
import numpy as np
from training.experiment_calibration import feature_rows, fit_linear, threshold, apply_rule


class CalibrationContracts(unittest.TestCase):
    def test_permutation_and_tied_relevant_candidates(self):
        row={'query':'related settings','candidates':[{'id':'a'},{'id':'b'},{'id':'c'}]}
        rule={'features':['score','bestMargin'],'weights':[1.,1.],'bias':0.,'threshold':.8}
        tiers=[[(0,0,0)]*3]
        self.assertEqual(apply_rule([row],[[.9,.9,.1]],tiers,rule),[[0,1]])
        a=feature_rows([.9,.8,.2]);b=feature_rows([.2,.9,.8])
        np.testing.assert_allclose(a,b[[1,2,0]])

    def test_hard_tiers_and_cosine_sort_survive_rejection(self):
        row={'query':'profile'}
        tiers=[[(3,0,1),(2,1,.8),(0,0,0),(0,0,0)]]
        rule={'features':['score'],'weights':[-1.],'bias':0.,'threshold':-1.}
        self.assertEqual(apply_rule([row],[[0.,0.,.8,.9]],tiers,rule),[[0,1,3,2]])

    def test_zero_or_invalid_embeddings_never_promote(self):
        rule={'features':['score'],'weights':[1.],'bias':100.,'threshold':0.}
        self.assertEqual(apply_rule([{'query':'coworkers'}],[[-np.inf,np.nan]],[[(0,0,0)]*2],rule),[[]])
        self.assertEqual(apply_rule([{'query':'x'}],[[.99]],[[(0,0,0)]],rule),[[]])

    def test_threshold_controls_no_match_ties(self):
        rows=[{'relevance':{'a':0}}]*20;values=[[.9]]+[[.5]]*19
        gate=threshold(rows,values,.05)
        self.assertGreater(gate,.5);self.assertEqual(sum(v[0]>=gate for v in values),1)

    def test_single_candidate_has_no_artificial_gap(self):
        np.testing.assert_allclose(feature_rows([.7]),[[.7,0.,0.,0.]])

    def test_fitter_uses_labels_and_returns_finite_small_rule(self):
        x=np.array([[-1.],[-.8],[.8],[1.]]);y=np.array([0.,0.,1.,1.])
        a=fit_linear(x,y,1.);b=fit_linear(x,1-y,1.)
        self.assertGreater(a['weights'][0],0);self.assertLess(b['weights'][0],0)
        self.assertTrue(np.isfinite([a['bias'],*a['weights']]).all())


if __name__=='__main__':unittest.main()
