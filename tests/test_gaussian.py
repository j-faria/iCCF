import numpy as np

def test_gaussfit():
    from iCCF import gaussfit
    from iCCF.utils import load_example_data
    x, y = load_example_data()
    result = gaussfit(x, y)
    known_result = np.array([-1.1951676088e+06,  3.5308442761e+00,  2.8687976305e+00,
                             3.0551781132e+06])
    np.testing.assert_allclose(result, known_result)
