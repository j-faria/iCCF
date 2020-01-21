from collections import Iterable

import numpy as np
import matplotlib.pyplot as plt
from os.path import basename
from glob import glob
import math
import warnings
from astropy.io import fits
from cached_property import cached_property

from .gaussian import gauss, gaussfit, FWHM as FWHMcalc, RV, RVerror, contrast
from .bisector import BIS, BIS_HARPS as BIS_HARPS_calc, bisector
from .vspan import vspan
from .wspan import wspan
from .keywords import getRVarray, getBJD
from . import writers


EPS = 1e-5 # all indicators are accurate up to this epsilon
nEPS = abs(math.floor(math.log(EPS, 10))) # number of decimals for output


def rdb_names(names):
    """ Return the usual .rdb format names. """
    r = []
    for name in names:
        # print(name)
        if name.lower() in ('rv', 'vrad', 'radvel'):
            r.append('vrad')
        elif name.lower() in ('rve', 'svrad', 'error', 'err', 'rverror'):
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
    """ Class to hold CCF indicators """
    def __init__(self, rv, ccf, RV_on=True, FWHM_on=True, BIS_on=True,
                 Vspan_on=True, Wspan_on=True, contrast_on=True,
                 BIS_HARPS=False):
        """
        The default constructor takes `rv` and `ccf` arrays as input, see
        `Indicators.from_file` for another way to create the object from a CCF
        fits file. Keyword parameters turn specific indicators on or off.
        Is `BIS_HARPS` is True, the BIS is calculated using the same routine as
        in the HARPS pipeline.
        """
        self.rv = rv
        self.ccf = ccf
        self.filename = None
        self.on_indicators = []
        if RV_on: 
            self.on_indicators.append('RV')
            self.on_indicators.append('RVerror')
        if FWHM_on: self.on_indicators.append('FWHM')
        if contrast_on: self.on_indicators.append('contrast')
        if BIS_on: self.on_indicators.append('BIS')
        if Vspan_on: self.on_indicators.append('Vspan')
        if Wspan_on: self.on_indicators.append('Wspan')
        self.on_indicators_rdb = rdb_names(self.on_indicators)

        self._use_bis_from_HARPS = BIS_HARPS

        self._EPS = EPS
        self._nEPS = nEPS

    def __repr__(self):
        if self.filename is None:
            r = f'CCFindicators(RVmin={self.rv.min()}; '\
                f'RVmax={self.rv.max()}; size={self.rv.size})'
        else:
            r = f'CCFindicators(CCF from {basename(self.filename)})'
        return r

    def __len__(self):
        return 1


    @classmethod
    def from_file(cls, file, hdu_number=1, data_index=-1, sort_bjd=True,
                  **kwargs):
        """ 
        Create an `Indicators` object from one or more fits files.
        
        Parameters
        ----------
        file : str or list of str
            The name(s) of the fits file(s)
        hdu_number : int, default = 1
            The index of the HDU list which contains the CCF
        data_index : int, default = -1
            The index of the .data array which contains the CCF. The data will 
            be accessed as ccf = HDU[hdu_number].data[data_index,:]
        sort_bjd : bool
            If True (default) and filename is a list of files, sort them by BJD
            before reading
        """

        if '*' in file or '?' in file:
            file = glob(file)

        # list of files
        if isinstance(file, Iterable) and not isinstance(file, str): 
            if sort_bjd:
                file = sorted(file, key=getBJD)
            return [cls.from_file(f) for f in file]

            # N = len(file)
            # rv, ccf = [], []
            # for i in range(N):
            #     f = file[i]
            #     rv.append(getRVarray(f))
            #     hdul = fits.open(f)
            #     ccf.append(hdul[hdu_number].data[data_index, :])
        
        # just one file
        elif isinstance(file, str):
            # one file only
            rv = getRVarray(file)
            hdul = fits.open(file)
            ccf = hdul[hdu_number].data[data_index, :]
        
            I = cls(rv, ccf, **kwargs)
            I.filename = file
            I.HDU = hdul
            return I

        else:
            raise ValueError(
                'Input to `from_file` should be a string or list of strings.')


    @cached_property
    def bjd(self):
        return getBJD(self.filename, mjd=False)


    @cached_property
    def RV(self):
        return RV(self.rv, self.ccf)

    @cached_property
    def RVerror(self):
        try:
            eccf = self.HDU[2].data[-1,:] # for ESPRESSO
        except Exception as e:
            warnings.warn(e)
            warnings.warn('Cannot access CCF uncertainties, using 1.0.')
            eccf = np.ones_like(self.rv)
        finally:
            return RVerror(self.rv, self.ccf, eccf)


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
    def contrast(self):
        return contrast(self.rv, self.ccf)

    @property
    def all(self):
        return tuple(self.__getattribute__(i) for i in self.on_indicators)

    def to_dict(self):
        return writers.to_dict(self)
    
    def to_rdb(self, filename='stdout', clobber=False):
        return writers.to_rdb(self, filename, clobber)


    def plot(self, show_bisector=False):
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(self.rv, self.ccf, 's-', ms=3)
        if show_bisector:
            out = BIS(self.rv, self.ccf, full_output=True)
            # BIS, c, bot, ran, \
            #    (bottom_limit1, bottom_limit2), (top_limit1, top_limit2), \
            #    fl1, fl2, bisf
            bisf = out[-1]
            top_limit = out[-4][1]
            bot_limit = out[-5][0]
            yy = np.linspace(bot_limit, top_limit, 100)
            ax.plot(bisf(yy), yy, 'k')
        ax.set(xlabel='RV [km/s]', ylabel='CCF')
        plt.show()


def indicators_from_files(files, rdb_format=True, show=True, show_bjd=True,
                          sort_bjd=True, **kwargs):

    if sort_bjd:
        files = sorted(files, key=getBJD)

    for j, f in enumerate(files):
        if show_bjd:
            bjd = getBJD(f)

        I = Indicators.from_file(f, **kwargs)
        if j == 0 and show:
            if rdb_format:
                lst = (['jdb'] + I.on_indicators_rdb) if show_bjd \
                    else I.on_indicators_rdb
                print('\t'.join(lst))
                print('\t'.join([len(s) * '-' for s in lst]))
            else:
                if show_bjd:
                    print(['jdb'] + I.on_indicators)
                else:
                    print(I.on_indicators)

        if rdb_format:
            print(
                '\t'.join([f'{bjd:<.6f}'] + [f'{ind:<.5f}' for ind in I.all]))
        else:
            print((bjd, ) + I.all)
