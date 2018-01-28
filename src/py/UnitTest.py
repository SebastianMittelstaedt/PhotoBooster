import unittest
import numpy as np
from src.py.ColorConversion import ColorConversion

class ColorConversionTest(unittest.TestCase):

    def test_rgb2xyz(self):
        rgb = [255.0,255.0,255.0]
        xyz = ColorConversion.rgb2xyz_cpu(rgb)
        rgb2 = np.round(ColorConversion.xyz2rgb_cpu(xyz),0).tolist()
        rgb = np.round(rgb,0).tolist()
        self.assertEqual(rgb2,rgb)

    def test_rgb2lms(self):
        rgb = [255.0,231.0,4.0]
        lms = ColorConversion.rgb2lms_cpu(rgb)
        rgb2 = np.round(ColorConversion.lms2rgb_cpu(lms),0).tolist()
        rgb = np.round(rgb,0).tolist()
        self.assertEqual(rgb2,rgb)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ColorConversionTest)
    unittest.TextTestRunner().run(suite)