from collections.abc import Iterable
from functools import lru_cache
import os

import numpy as np
import matplotlib.pyplot as plt
from os.path import basename
from glob import glob
import math
import warnings

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from .gaussian import gauss, gaussfit, FWHM as FWHMcalc, RV, RVerror, contrast
from .bisector import BIS, BIS_HARPS as BIS_HARPS_calc, BIS_ESP as BIS_ESP_calc
from .vspan import vspan
from .wspan import wspan
from .keywords import (getRV, getRVerror, getFWHM, getBIS, getBJD, getMASK, getRVarray,
                       getINSTRUMENT, getOBJECT)
from . import writers
from .ssh_files import ssh_fits_open
from .utils import no_stack_warning, one_line_warning

EPS = 1e-5  # all indicators are accurate up to this epsilon
nEPS = abs(math.floor(math.log(EPS, 10)))  # number of decimals for output


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
    """ Class to hold CCFs and CCF indicators """
    def __init__(self, rv, ccf, eccf=None, RV_on=True, FWHM_on=True,
                 BIS_on=True, Vspan_on=True, Wspan_on=True, contrast_on=True,
                 BIS_HARPS=False):
        """
        The default constructor takes `rv` and `ccf` arrays as input. See
        `Indicators.from_file` for another way to create the object from a CCF
        fits file. The CCF uncertainties can be provided as the `eccf` array.
        Keyword parameters turn specific indicators on or off. Is `BIS_HARPS` 
        is True, the BIS is calculated using the same routine as in the HARPS 
        pipeline.
        """
        self.rv = rv
        self.ccf = ccf
        self.eccf = eccf
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

        # the ESPRESSO pipeline does not consider the errors when fitting the
        # CCF, so by default we do not use them as well
        self._use_errors = False

        self._use_bis_from_HARPS = BIS_HARPS

        self.HDU = None
        self._EPS = EPS
        self._nEPS = nEPS

    def __repr__(self):
        if self.filename is None:
            info = f'RVmin={self.rv.min()}; RVmax={self.rv.max()}; n={self.rv.size}'
            r = f'CCFindicators({info})'
        else:
            r = f'CCFindicators(CCF from {basename(self.filename)})'
        return r

    def __len__(self):
        return 1

    @classmethod
    def from_file(cls, file, hdu_number=1, data_index=-1, sort_bjd=True, **kwargs):
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
        sort_bjd : bool (default True)
            If True and filename is a list of files, sort them by BJD before 
            reading
        """

        if isinstance(file, list) and len(file) == 1:
            file = file[0]

        if '*' in file or '?' in file:
            file = glob(file)

        # list of files
        if isinstance(file, Iterable) and not isinstance(file, str):
            indicators = []
            for f in tqdm(file):
                try:
                    indicators.append(
                        cls.from_file(f, hdu_number, data_index, sort_bjd, **kwargs))
                except Exception as e:
                    print(f'ERROR reading "{f}"')
                    print(e)

            if sort_bjd:
                return sorted(indicators, key=lambda i: i.bjd)
            else:
                return indicators

        # just one file
        elif isinstance(file, str):

            # for notccf in ('S1D', 'S2D'):
            #     if notccf in file:
            #         msg = f"filename {file} contains '{notccf}', "\
            #                "are you sure it contains a CCF?"
            #         no_stack_warning(msg)

            user, host = kwargs.pop('USER', None), kwargs.pop('HOST', None)
            port = kwargs.pop('port', 22)
            verbose = kwargs.pop('verbose', False)

            rv, hdul = getRVarray(file, return_hdul=True, USER=user, HOST=host)

            if len(hdul) == 1 and hdu_number == 1:
                warnings.warn('file only has one HDU, using hdu_number=0')
                hdu_number = 0

            ccf = hdul[hdu_number].data[data_index, :]
            try:
                eccf = hdul[hdu_number + 1].data[data_index, :]
            except IndexError:
                eccf = None
                warnings.warn('no CCF errors found')

            I = cls(rv, ccf, eccf=eccf, **kwargs)

            # save attributes
            I.filename = file
            I._SCIDATA = hdul[hdu_number].data
            try:
                I._ERRDATA = hdul[2].data
                I._QUALDATA = hdul[3].data
            except IndexError:
                pass
            I.HDU = hdul
            hdul.close()
            I._hdu_number = hdu_number
            I._data_index = data_index

            return I

        else:
            raise ValueError(
                'Input to `from_file` should be a string or list of strings.')

    @property
    def norders(self):
        return self._SCIDATA.shape[0] - 1

    @property
    def OBJECT(self):
        return getOBJECT(self.filename, hdul=self.HDU)

    @property
    # @lru_cache
    def bjd(self):
        """ Barycentric Julian Day when the observation was made """
        return getBJD(self.filename, hdul=self.HDU, mjd=False)

    @property
    def mask(self):
        """ Mask used for the CCF calculation """
        return getMASK(self.filename, hdul=self.HDU)

    @property
    def instrument(self):
        return getINSTRUMENT(self.filename, hdul=self.HDU)

    @property
    def RV(self):
        """ The measured radial velocity, from a Gaussian fit to the CCF [km/s] """
        eccf = self.eccf if self._use_errors else None
        try:
            return RV(self.rv, self.ccf, eccf, guess_rv=self.pipeline_RV)
        except ValueError:
            return RV(self.rv, self.ccf, eccf)

    @property
    def RVerror(self):
        """ Photon-noise uncertainty on the measured radial velocity [km/s] """
        if self.eccf is not None:  # CCF uncertainties were provided
            if self.eccf.size != self.ccf.size:
                raise ValueError('CCF and CCF errors not of the same size')
            eccf = self.eccf
        else:  # try reading it from the HDU
            try:
                eccf = self._ERRDATA[-1, :]  # for ESPRESSO
            except Exception:
                # warnings.warn('Cannot access CCF uncertainties, looking for value in header')
                try:
                    rve = getRVerror(None, hdul=self.HDU)
                    return rve
                except ValueError:
                    # warnings.warn('Cannot access CCF uncertainties, using 1.0')
                    eccf = np.ones_like(self.rv)

        return RVerror(self.rv, self.ccf, eccf)

    @property
    # @lru_cache
    def individual_RV(self):
        """ Individual radial velocities for each spectral order """
        if not hasattr(self, 'HDU'):
            raise ValueError('Cannot access individual CCFs (no HDU attribute)')
        from .gaussian import fwhm2sig
        RVs = []
        for ccf in self._SCIDATA[:-1]:
            if np.nonzero(ccf)[0].size == 0:
                RVs.append(np.nan)
            else:
                p0 = [-np.ptp(ccf), self.RV, fwhm2sig(self.FWHM), ccf.mean()]
                RVs.append(RV(self.rv, ccf, p0=p0))

        return np.array(RVs)

    @property
    # @lru_cache
    def individual_RVerror(self):
        """ Individual radial velocity errors for each spectral order """
        if not hasattr(self, 'HDU'):
            raise ValueError(
                'Cannot access individual CCFs (no HDU attribute)')

        if not hasattr(self, '_ERRDATA'):
            return np.full_like(self.individual_RV, np.nan)

        RVes = []
        CCFs, eCCFs = self._SCIDATA, self._ERRDATA
        for ccf, eccf in zip(CCFs[:-1], eCCFs[:-1]):
            if np.nonzero(ccf)[0].size == 0:
                RVes.append(np.nan)
            else:
                RVes.append(RVerror(self.rv, ccf, eccf))

        return np.array(RVes)

    @property
    def FWHM(self):
        """ The full width at half maximum of the CCF [km/s] """
        eccf = self.eccf if self._use_errors else None
        try:
            return FWHMcalc(self.rv, self.ccf, eccf,
                            guess_rv=self.pipeline_RV)
        except ValueError:
            return FWHMcalc(self.rv, self.ccf, eccf)

    @property
    def FWHMerror(self):
        """ Photon noise uncertainty on the FWHM of the CCF [km/s] """
        warnings.formatwarning = one_line_warning
        warnings.warn('FWHMerror is currently just 2 x RVerror')
        return 2.0 * self.RVerror

    @property
    def BIS(self):
        """ Bisector span [km/s] """
        warnings.warn('BIS currently does NOT match the ESPRESSO pipeline')
        if self._use_bis_from_HARPS:
            return BIS_HARPS_calc(self.rv, self.ccf)
        else:
            return BIS_ESP_calc(self.rv, self.ccf)[0]
            # return BIS(self.rv, self.ccf)

    @property
    def BISerror(self):
        """ Photon noise uncertainty on the BIS of the CCF [km/s] """
        warnings.warn('BISerror is currently just 2 x RVerror')
        return 2.0 * self.RVerror


    @property
    def Vspan(self):
        """ Vspan indicator [km/s], see Boisse et al. (2011b, A&A 528 A4) """
        return vspan(self.rv, self.ccf)

    @property
    def Wspan(self):
        """ Wspan indicator [km/s], see Santerne et al. (2015, MNRAS 451, 3) """
        return wspan(self.rv, self.ccf)

    @property
    def contrast(self):
        """ The contrast (depth) of the CCF, measured in percentage """
        return contrast(self.rv, self.ccf, self.eccf)

    @property
    def contrast_error(self):
        """ Uncertainty on the contrast (depth) of the CCF, measured in percentage """
        return contrast(self.rv, self.ccf, self.eccf, error=True)


    @property
    def all(self):
        """ All the indicators that are on """
        return tuple(self.__getattribute__(i) for i in self.on_indicators)

    @property
    def pipeline_RV(self):
        """
        The radial velocity as derived by the pipeline, from the header.
        """
        if not hasattr(self, 'HDU') or self.HDU is None:
            raise ValueError('Cannot access header (no HDU attribute)')

        return getRV(None, hdul=self.HDU)

    @property
    def pipeline_FWHM(self):
        """
        The FWHM as derived by the pipeline, from the header.
        """
        if not hasattr(self, 'HDU'):
            raise ValueError('Cannot access header (no HDU attribute)')

        return getFWHM(None, hdul=self.HDU)

    @property
    def pipeline_BIS(self):
        """
        The BIS SPAN as derived by the pipeline, from the header.
        """
        if not hasattr(self, 'HDU'):
            raise ValueError('Cannot access header (no HDU attribute)')

        return getBIS(None, hdul=self.HDU)


    @property
    def pipeline_RVerror(self):
        """ 
        The RV error as derived by the pipeline, from the header.
        """
        if not hasattr(self, 'HDU'):
            raise ValueError('Cannot access header (no HDU attribute)')

        return getRVerror(None, hdul=self.HDU)

    def recalculate_ccf(self, orders=None, weighted=False):
        """
        Recompute the final CCF from the CCFs of individual orders

        Args:
            orders (slice, int, tuple, list, array): 
                List of orders for which to sum the CCFs. If None, use all orders. 
                If an int, return directly the CCF of that order (1-based).
        """
        if orders is None:
            return self.ccf

        warnings.warn('In this function, orders are 1-based. Make sure the right orders are being used!')
        print(orders - 1)

        if isinstance(orders, int):
            return self._SCIDATA[orders - 1]
        else:
            if weighted:
                has_errors = np.where([(self._ERRDATA[j] != 0).all() for j in range(self.norders)])[0]
                orders = np.array(orders)[np.isin(orders, has_errors)]
                d = self._SCIDATA[orders - 1]
                e = self._ERRDATA[orders - 1]
                return np.average(d, axis=0, weights=1/e**2) * len(orders)
            else:
                return self._SCIDATA[orders - 1].sum(axis=0)

    def remove_orders(self, orders, weighted=False):
        """
        Remove specific orders and recompute the CCF

        Args:
            orders (slice, int, tuple, list, array): 
                List of orders for which to sum the CCFs. If None, use all orders. 
                If an int, return directly the CCF of that order (1-based).
        """
        previous_RV = self.RV
        all_orders = np.arange(self.norders)
        orders = np.delete(all_orders, orders)
        print('Recalculating CCF for subset of orders')
        self.ccf = self.recalculate_ccf(orders, weighted=weighted)
        if self.RV != previous_RV:
            print('RV changed from', previous_RV)
            print('             to', self.RV, 'km/s')


    def reset(self):
        """ Reset the CCF to the original CCF """
        self.ccf = self._SCIDATA[-1]


    def check(self, verbose=False):
        """
        Check if the calculated RV, RVerror, and FWHM match the pipeline values
        """
        Q = {
            'RV': (self.RV, self.pipeline_RV),
            'RVerror': (self.RVerror, self.pipeline_RVerror),
            'FWHM': (self.FWHM, self.pipeline_FWHM),
        }
        for q, (val1, val2) in Q.items():
            try:
                if verbose:
                    print(f'comparing {q:8s}: calculated/pipeline:', end=' ')
                    print(f'{val1:.{self._nEPS}f} / {val2:.{self._nEPS}f}', end=' ')
                np.testing.assert_almost_equal(val1, val2, self._nEPS, err_msg='')
                if verbose:
                    print('✓')
            except ValueError as e:
                no_stack_warning(str(e))

        # try:
        #     val1, val2 = self.RVerror, self.pipeline_RVerror
        #     if verbose:
        #         print('comparing RVerror calculated/pipeline:', end=' ')
        #         print(f'{val1:.{self._nEPS}f} / {val2:.{self._nEPS}f}')
        #     np.testing.assert_almost_equal(val1, val2, self._nEPS, err_msg='')
        # except ValueError as e:
        #     no_stack_warning(str(e))

        # try:
        #     val1, val2 = self.FWHM, self.pipeline_FWHM
        #     if verbose:
        #         print('comparing FWHM calculated/pipeline:', end=' ')
        #         # print(f'{val1:.{2}f} / {val2:.{2}f}')
        #         # no_stack_warning(
        #         #     'As of now, FWHM is only compared to 2 decimal places')
        #         print(f'{val1:.{self._nEPS}f} / {val2:.{self._nEPS}f}')
        #     np.testing.assert_almost_equal(val1, val2, self._nEPS, err_msg='')
        # except ValueError as e:
        #     no_stack_warning(str(e))

        return True  # all checks passed!

    @property
    def zero_ccfs(self):
        self._SCIDATA

    def to_dict(self):
        return writers.to_dict(self)

    def to_rdb(self, filename='stdout', clobber=False):
        return writers.to_rdb(self, filename, clobber)

    def plot(self, ax=None, show_fit=True, show_bisector=False, over=0):
        """ Plot the CCF, together with the Gaussian fit and the bisector """
        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)

        label = os.path.basename(self.filename)
        eccf = self.eccf if self._use_errors else None

        ax.errorbar(self.rv, self.ccf, eccf, fmt='s-', ms=3, label=label)

        if show_fit:
            r = np.ptp(self.rv)
            vv = np.linspace(self.rv.min() - over * r, self.rv.max() + over * r, 1000)

            try:
                pfit = gaussfit(self.rv, self.ccf, yerr=eccf,
                                guess_rv=self.pipeline_RV)
            except ValueError:
                pfit = gaussfit(self.rv, self.ccf, yerr=eccf)

            ax.plot(vv, gauss(vv, pfit),
                    label=f'Gaussian fit (RV={self.RV:.3f} km/s)')

        if show_bisector:
            out = BIS(self.rv, self.ccf, full_output=True)
            bisf = out[-1]
            top_limit = out[-4][1]
            bot_limit = out[-5][0]
            yy = np.linspace(bot_limit, top_limit, 25)
            ax.plot(bisf(yy), yy, '-s',
                    label=f'Bisector (BIS={self.BIS*1e3:.3f} m/s)')

        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.15))
        ax.set(xlabel='RV [km/s]', ylabel='CCF')
        return ax.figure, ax

    def plot_individual_CCFs(self, ax=None, show_errors=True, **kwargs):
        """ Plot the CCFs for individual orders """
        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)
        
        n = self.individual_RV.size
        for i in range(1, n + 1):
            _x = (self.rv - self.rv[0]) / np.ptp(self.rv) - 0.5 + i
            if show_errors:
                ax.errorbar(_x, self._SCIDATA[i - 1], self._ERRDATA[i - 1], **kwargs)
            else:
                ax.plot(_x, self._SCIDATA[i - 1], **kwargs)

        ax.set(xlabel='spectral order', ylabel='CCF', xlim=(-5, n+5))
        return ax.figure, ax

    def plot_individual_RV(self, ax=None, relative=False, **kwargs):
        """ Plot the RV for individual orders """
        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)

        n = self.individual_RV.size
        orders = np.arange(1, n + 1)
        if relative:
            ax.errorbar(orders, (self.individual_RV - self.RV) / self.individual_RVerror,
                        self.individual_RVerror, fmt='o', label='individual order RV')
            ax.axhline(0.0, color='darkgreen', ls='--', label='final RV')
        else:
            ax.errorbar(orders, self.individual_RV,
                        self.individual_RVerror, fmt='o', label='individual order RV')
            ax.axhline(self.RV, color='darkgreen', ls='--', label='final RV')
            nans = np.isnan(self.individual_RV)
            if nans.any():
                ax.plot(orders[nans], np.full(n, self.RV)[nans], 'rx', label='NaN')

        # m = np.full_like(orders, self.RV, dtype=float)
        # e = np.full_like(orders, self.RVerror, dtype=float)
        # ax.fill_between(orders, m-e, m+e, color='g', alpha=0.3)

        if kwargs.get('legend', True):
            ax.legend()

        if relative:
            ax.set(xlabel='spectral order', ylabel='RV - final RV [m/s]', xlim=(-5, n+5))
        else:
            ax.set(xlabel='spectral order', ylabel='RV [m/s]', xlim=(-5, n+5))

        return ax.figure, ax


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
            print('\t'.join([f'{bjd:<.6f}'] + [f'{ind:<.5f}'
                                               for ind in I.all]))
        else:
            print((bjd, ) + I.all)
