import unittest
import numpy as np
from training.export import pack,unpack,quantize
class ExportTests(unittest.TestCase):
    def test_int6_extrema_and_nonbyte_counts(self):
        for count in range(1,18):
            values=np.resize(np.array([-31,0,31,-1,1],dtype=np.int8),count)
            np.testing.assert_array_equal(values,unpack(pack(values,6),count,6))
    def test_int8_extrema(self):
        a=np.array([-127,0,127],dtype=np.int8);np.testing.assert_array_equal(a,unpack(pack(a,8),3,8))
    def test_invalid_codes_and_padding(self):
        with self.assertRaises(ValueError):unpack(bytes([63]),1,6)
        with self.assertRaises(ValueError):unpack(bytes([255]),1,6)
        with self.assertRaises(ValueError):pack(np.array([32]),6)
        with self.assertRaises(ValueError):unpack(bytes([0,0]),1,6)
    def test_symmetric_rounding_and_zero(self):
        q,s=quantize(np.array([127, .5,-.5,1.5,-1.5]),8);self.assertEqual(q.tolist(),[127,1,-1,2,-2])
        q,s=quantize(np.zeros(3),6);self.assertEqual(float(s),1);self.assertEqual(q.tolist(),[0,0,0])
    def test_nonfinite(self):
        with self.assertRaises(ValueError):quantize(np.array([float('nan')]),8)
if __name__=='__main__':unittest.main()
