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

def test_read_spectral_format():
    from iCCF.chromatic import read_spectral_format
    _ = read_spectral_format()

def test_find_mask():
    from iCCF.utils import find_data_file

    try:
        find_data_file('ESPRESSO_G9.fits')
    except FileNotFoundError as e:
        assert False, f'Data file not found: {e}'

    with pytest.raises(FileNotFoundError):
        find_data_file('ESPRESSO_G99.fits')