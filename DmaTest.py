import unittest 
import numpy as np 

from moment_r2dp import DynamicMomentR2DP 


class DmaTest(unittest.TestCase):


    def setUp(self):

        self.processor = DynamicMomentR2DP(5)



    def test_get_l1_Gaussian(self):

        result = self.processor.get_l1_Gaussian(sigma=1.0, t=1)
        self.assertAlmostEqual(result, 1.0)

        result = self.processor.get_l1_Gaussian(sigma=2.0, t=4)
        self.assertAlmostEqual(result, 4.0)

        result = self.processor.get_l1_Gaussian(sigma=2.0, t=9)
        self.assertAlmostEqual(result, 6.0)

    def test_get_epsilon_gaussian(self):

        result = self.processor.get_epsilon_gaussian(time=1, sigma=1, delta=0.1)
        expected = (1 + 2 * 1 * np.sqrt(2 * 1 * np.log(10)))/ (2* 1**2)
        self.assertAlmostEqual(result, expected, places=5)


    def test_get_optimum_sigma_gaussian(self):
        result = self.processor.get_optimum_sigma_gaussian(time=1, epsilon_bound=1, delta=0.1)
        self.assertTrue(result > 0)

    def test_M(self):
        k = 3
        theta = 0.5
        alpha = 2
        expected_result = np.power((1 - alpha * theta), -k)
        result = self.processor.M(k, theta, alpha)
        self.assertEqual(result, expected_result, "The M function did not return the expected result.")




if __name__=="__main__":
    unittest.main()