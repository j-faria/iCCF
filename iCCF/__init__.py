"""
Analysis tools for common line profile indicators measured from the
cross-correlation function (CCF).
"""
__all__ = ['Indicators', 'gauss', 'gaussfit']

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("iCCF")
except PackageNotFoundError: # package is not installed
    pass

from .bisector import BIS, BISplus, BISminus, BIS_HARPS
from .gaussian import gauss, gaussfit, RV, FWHM, contrast
from .bigaussian import bigauss, bigaussfit
from .vspan import vspan
from .wspan import wspan
from .meta import calculate_ccf
from .masks import Mask

from .chromatic import chromaticRV

# from .iCCF import EPS, nEPS

from .iCCF import Indicators
from_file = Indicators.from_file

from .iCCF import bjd, vrad, svrad