from collections import Iterable

import numpy as np
from astropy.io import fits
from cached_property import cached_property

from .gaussian import gauss, gaussfit, FWHM as FWHMcalc, RV
from .bisector import BIS, BIS_HARPS as BIS_HARPS_calc
from .vspan import vspan
from .wspan import wspan
from .keywords import getRVarray, getBJD


def rdb_names(names):
    """ Return the usual .rdb format names. """
    r = []
    for name in names:
        if name.lower() in ('rv', 'vrad', 'radvel'):
            r.append('vrad')
        elif name.lower() in ('rve', 'svrad', 'error', 'err'):
            r.append('svrad')
        elif name.lower() in ('fwhm'):
            r.append('fwhm')
        elif name.lower() in ('bis', 'bisspan', 'bis_span'):
            r.append('bis_span')
        elif name.lower() in ('contrast'):
            r.append('contrast')
        else:
            r.append(name)
    return r


class Indicators:
    def __init__(self, rv, ccf, RV_on=True, FWHM_on=True, BIS_on=True,
                 Vspan_on=True, Wspan_on=True, BIS_HARPS=False):
        self.rv = rv
        self.ccf = ccf
        self.filename = None
        self.on_indicators = []
        if RV_on: self.on_indicators.append('RV')
        if FWHM_on: self.on_indicators.append('FWHM')
        if BIS_on: self.on_indicators.append('BIS')
        if Vspan_on: self.on_indicators.append('Vspan')
        if Wspan_on: self.on_indicators.append('Wspan')
        self.on_indicators_rdb = rdb_names(self.on_indicators)

        self._use_bis_from_HARPS = BIS_HARPS

    @classmethod
    def from_file(cls, filename, hdu_number=0, data_index=-1, **kwargs):
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

        I = cls(rv, ccf, **kwargs)
        I.filename = filename

        return I

    @cached_property
    def RV(self):
        return RV(self.rv, self.ccf)

    @cached_property
    def FWHM(self):
        return FWHMcalc(self.rv, self.ccf)

    @cached_property
    def BIS(self):
        if self._use_bis_from_HARPS:
            return BIS_HARPS_calc(self.rv, self.ccf)
        else:
            return BIS(self.rv, self.ccf)

    @cached_property
    def Vspan(self):
        return vspan(self.rv, self.ccf)

    @cached_property
    def Wspan(self):
        return wspan(self.rv, self.ccf)

    @cached_property
    def all(self):
        return tuple(self.__getattribute__(i) for i in self.on_indicators)


def indicators_from_files(files, rdb_format=True, show=True, show_bjd=True,
                          **kwargs):

    for j, f in enumerate(files):
        if show_bjd:
            bjd = getBJD(f)

        I = Indicators.from_file(f, **kwargs)
        if j == 0 and show:
            if rdb_format:
                lst = (['bjd'] + I.on_indicators_rdb) if show_bjd \
                    else I.on_indicators_rdb
                print('\t'.join(lst))
                print('\t'.join([len(s) * '-' for s in lst]))
            else:
                if show_bjd:
                    print(['bjd'] + I.on_indicators)
                else:
                    print(I.on_indicators)

        if rdb_format:
            print(
                '\t'.join([f'{bjd:<.6f}'] + [f'{ind:<.5f}' for ind in I.all]))
        else:
            print((bjd,) + I.all)
