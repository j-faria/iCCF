import pytest

import iCCF
from iCCF.utils import load_example_data

def test_Indicators():
    x, y = load_example_data()
    i = iCCF.Indicators(x, y)

@pytest.fixture
def indicator():
    x, y = load_example_data()
    i = iCCF.Indicators(x, y)
    return i

def test_properties(indicator):
    i = indicator
    i.RV
    i.FWHM
