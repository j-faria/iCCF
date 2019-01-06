from collections import Iterable

import numpy as np
from astropy.io import fits

from .gaussian import gauss, gaussfit, FWHM as FWHMcalc
from .bisector import BIS, BIS_HARPS as BIS_HARPS_calc
from .vspan import vspan
from .wspan import wspan
from .keywords import getRVarray


class Indicators:
    def __init__(self, rv, ccf, FWHM_on=True, BIS_on=True, Vspan_on=True,
                 Wspan_on=True, BIS_HARPS=False):
        self.rv = rv
        self.ccf = ccf
        self.filename = None
        self.on_indicators = []
        if FWHM_on: self.on_indicators.append('FWHM')
        if BIS_on: self.on_indicators.append('BIS')
        if Vspan_on: self.on_indicators.append('Vspan')
        if Wspan_on: self.on_indicators.append('Wspan')
        
        self._use_bis_from_HARPS = BIS_HARPS


    @classmethod
    def from_file(cls, filename, hdu_number=0, data_index=-1):
        """ 
        Create an `Indicators` object from one or more fits files.
        
        Parameters
        ----------
        filename : str or list of str
            The name(s) of the fits file(s)
        hdu_number : int, default = 0
            The index of the HDU list which contains the CCF
        data_index : int, default = -1
            The index of the .data array which contains the CCF. The data will 
            be accessed as ccf = HDU[hdu_number].data[data_index,:]
        """
        if isinstance(filename, Iterable) and not isinstance(filename, str):
            # list of files
            N = len(filename)
            rv, ccf = [], []
            for i in range(N):
                f = filename[i]
                rv.append(getRVarray(f))
                hdul = fits.open(f)
                ccf.append(hdul[hdu_number].data[data_index, :])

        if isinstance(filename, str):
            # one file only
            rv = getRVarray(filename)
            hdul = fits.open(filename)
            ccf = hdul[hdu_number].data[data_index, :]
        else:
            raise ValueError(
                'Input to `from_file` should be a string or list of strings.')

        I = cls(rv, ccf)
        I.filename = filename

        return I

    @property
    def FWHM(self):
        return FWHMcalc(self.rv, self.ccf)

    @property
    def BIS(self):
        if self._use_bis_from_HARPS:
            return BIS_HARPS_calc(self.rv, self.ccf)
        else:
            return BIS(self.rv, self.ccf)

    @property
    def Vspan(self):
        return vspan(self.rv, self.ccf)

    @property
    def Wspan(self):
        return wspan(self.rv, self.ccf)

    @property
    def all(self):
        return tuple(self.__getattribute__(i) for i in self.on_indicators)
