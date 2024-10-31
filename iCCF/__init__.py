"""
Analysis tools for common line profile indicators measured from the
cross-correlation function (CCF).
"""

from .version import __version__

from .bisector import BIS, BISplus, BISminus, BIS_HARPS
from .gaussian import gauss, gaussfit, RV, FWHM, contrast
from .bigaussian import bigauss, bigaussfit
from .vspan import vspan
from .wspan import wspan
from .meta import calculate_ccf
from . import utils

from .chromatic import chromaticRV

from .iCCF import EPS, nEPS
from .iCCF import Indicators
from_file = Indicators.from_file
