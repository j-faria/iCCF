from __future__ import annotations
from collections.abc import Iterable
import os

from os.path import basename
from glob import glob
import warnings

import numpy as np
from tqdm import tqdm

from .gaussian import gauss, gaussfit, FWHM as FWHMcalc, RV, RVerror, contrast
from .bisector import BIS, BIS_HARPS as BIS_HARPS_calc, BIS_ESP as BIS_ESP_calc
from .vspan import vspan
from .wspan import wspan
from .keywords import (getRVerror, getBJD, getRVarray, getINSTRUMENT)
from . import writers
from .utils import one_line_warning


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
    _warnings = True

    filename: str | None = None

    def __init__(self, rv, ccf, eccf=None, RV_on=True, FWHM_on=True,
                 BIS_on=True, Vspan_on=True, Wspan_on=True, contrast_on=True,
                 BIS_HARPS=False):
        """
        The default constructor takes `rv` and `ccf` arrays as input. See
        `Indicators.from_file` for another way to create the object from a CCF
        fits file. The CCF uncertainties can be provided as the `eccf` array.
        Keyword parameters turn specific indicators on or off. Is `BIS_HARPS` is
        True, the BIS is calculated using the same routine as in the HARPS
        pipeline.
        """
        self.rv = rv
        self.ccf = ccf
        self.eccf = eccf
        self.on_indicators = []
        if RV_on:
            self.on_indicators.append('RV')
            self.on_indicators.append('RVerror')
        if FWHM_on:
            self.on_indicators.append('FWHM')
        if contrast_on:
            self.on_indicators.append('contrast')
        if BIS_on:
            self.on_indicators.append('BIS')
        if Vspan_on:
            self.on_indicators.append('Vspan')
        if Wspan_on:
            self.on_indicators.append('Wspan')
        self.on_indicators_rdb = rdb_names(self.on_indicators)

        # the ESPRESSO pipeline does not consider the errors when fitting the
        # CCF, so by default we do not use them as well
        self._use_errors = False

        self._use_bis_from_HARPS = BIS_HARPS

        self.HDU = None

        # guess for RV in Gaussian fit
        self._guess_rv = None

    @property
    def _precision(self):
        return np.finfo(self.ccf.dtype).precision

    @property
    def _eps(self):
        return np.finfo(self.ccf.dtype).eps

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
    def from_file(cls, file, hdu_number=1, data_index=-1, sort_bjd=True,
                  keep_open=False, **kwargs):
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
        verbose = kwargs.pop('verbose', False)

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
                    if verbose:
                        print(f'ERROR reading "{f}"')
                    raise e

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

            I = cls(rv, ccf, eccf=eccf, **kwargs)  # noqa: E741

            # save attributes
            I.filename = file
            I._SCIDATA = hdul[hdu_number].data
            try:
                I._ERRDATA = hdul[2].data
                I._QUALDATA = hdul[3].data
            except IndexError:
                pass
            I.HDU = hdul

            if not keep_open:
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
        """ OBJECT keyword in the header """
        from .keywords import getOBJECT
        return getOBJECT(self.filename, hdul=self.HDU)

    @property
    # @lru_cache
    def bjd(self):
        """Barycentric Julian Day when the observation was made"""
        if self.filename is None:
            return None
        return getBJD(self.filename, hdul=self.HDU, mjd=False)

    @property
    def mask(self):
        """ Mask used for the CCF calculation """
        from .keywords import getMASK
        return getMASK(self.filename, hdul=self.HDU)

    @property
    def instrument(self):
        return getINSTRUMENT(self.filename, hdul=self.HDU)

    @property
    def guess_rv(self):
        if self._guess_rv is None:
            self._guess_rv = self.pipeline_RV
        return self._guess_rv

    @guess_rv.setter
    def guess_rv(self, value):
        self._guess_rv = value

    ############################################################################
    @property
    def RV(self):
        """ Radial velocity, from a Gaussian fit to the CCF [km/s] """
        eccf = self.eccf if self._use_errors else None
        try:
            return RV(self.rv, self.ccf, eccf, guess_rv=self._guess_rv)
        except ValueError:
            return RV(self.rv, self.ccf, eccf)

    @property
    def RVerror(self):
        """ Photon-noise uncertainty on the radial velocity [km/s] """
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
                except (ValueError, AttributeError):
                    # warnings.warn('Cannot access CCF uncertainties, using 1.0')
                    eccf = np.ones_like(self.rv)

        return RVerror(self.rv, self.ccf, eccf)

    ############################################################################
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
        if self._warnings:
            warnings.formatwarning = one_line_warning
            warnings.warn('FWHMerror is currently just 2 x RVerror')
        return 2.0 * self.RVerror

    ############################################################################
    @property
    def BIS(self):
        """ Bisector span [km/s] """
        # if self._warnings:
        #     warnings.formatwarning = one_line_warning
        #     warnings.warn('BIS currently does NOT match the ESPRESSO pipeline')
        if self._use_bis_from_HARPS:
            return BIS_HARPS_calc(self.rv, self.ccf)
        else:
            return BIS_ESP_calc(self.rv, self.ccf)[0]

    @property
    def BISerror(self):
        """ Photon noise uncertainty on the BIS of the CCF [km/s] """
        if self._warnings:
            warnings.formatwarning = one_line_warning
            warnings.warn('BISerror is currently just 2 x RVerror')
        return 2.0 * self.RVerror

    ############################################################################
    @property
    def Vspan(self):
        """ Vspan indicator [km/s], see Boisse et al. (2011b, A&A 528 A4) """
        return vspan(self.rv, self.ccf)

    @property
    def Wspan(self):
        """ Wspan indicator [km/s], see Santerne et al. (2015, MNRAS 451, 3) """
        return wspan(self.rv, self.ccf)

    @property
    def DeltaV(self):
        """
        DeltaV indicator from a biGaussian fit [km/s]
        see Figueira et al. (2013, A&A, 557, A93)
        """
        from .bigaussian import DeltaV

        eccf = self.eccf if self._use_errors else None
        return DeltaV(self.rv, self.ccf, eccf)

    def ΔV(self):
        """
        ΔV indicator from a biGaussian fit [km/s]
        see Figueira et al. (2013, A&A, 557, A93)
        """
        return self.DeltaV

    ############################################################################
    @property
    def contrast(self):
        """ The contrast (depth) of the CCF, measured in percentage """
        eccf = self.eccf if self._use_errors else None
        return contrast(self.rv, self.ccf, eccf)

    @property
    def contrast_error(self):
        """ Uncertainty on the contrast (depth) of the CCF, measured in percentage """
        return contrast(self.rv, self.ccf, self.eccf, error=True)

    ############################################################################
    def individual_value(self, function, needs_errors=False, **kwargs):
        if not hasattr(self, 'HDU'):
            raise ValueError('Cannot access individual CCFs (no HDU attribute)')

        if needs_errors:
            if not hasattr(self, '_ERRDATA'):
                return np.full_like(self.individual_RV, np.nan)

        vals = np.zeros(self.norders, dtype=self._SCIDATA.dtype)
        for i, ccf in enumerate(self._SCIDATA[:-1]):
            if np.count_nonzero(ccf) == 0:
                vals[i] = np.nan
            else:
                try:
                    if self._use_errors or needs_errors:
                        eccf = self._ERRDATA[i, :]
                    else:
                        eccf = None
                    vals[i] = function(self.rv, ccf, eccf, **kwargs)
                except RuntimeError:
                    vals[i] = np.nan
        return vals

    @property
    def individual_RV(self):
        """ Individual radial velocities for each spectral order [km/s] """
        return self.individual_value(RV)

    @property
    def individual_RVerror(self):
        """ Individual radial velocity errors for each spectral order [km/s] """
        return self.individual_value(RVerror, needs_errors=True)

    @property
    def individual_FWHM(self):
        """ Individual FWHM for each spectral order [km/s] """
        return self.individual_value(FWHMcalc)

    @property
    def individual_FWHMerror(self):
        """ Individual FWHM errors for each spectral order [km/s] """
        if self._warnings:
            warnings.formatwarning = one_line_warning
            warnings.warn('FWHMerror is currently just 2 x RVerror')
        return 2.0 * self.individual_RVerror

    @property
    def individual_contrast(self):
        """ The contrast (depth) of the CCF for each spectral order [%] """
        return self.individual_value(contrast)

    @property
    def individual_contrast_error(self):
        """ Uncertainty on the contrast (depth) of the CCF for each spectral order [%] """
        return self.individual_value(contrast, needs_errors=True, error=True)

    ############################################################################


    @property
    def all(self):
        """ All the indicators that are on """
        return tuple(self.__getattribute__(i) for i in self.on_indicators)

    def _check_for_HDU(self):
        if not hasattr(self, 'HDU') or self.HDU is None:
            raise ValueError('Cannot access header (no HDU attribute)')

    @property
    def pipeline_RV(self):
        """
        The radial velocity as derived by the pipeline, from the header.
        """
        self._check_for_HDU()
        from .keywords import getRV
        return getRV(None, hdul=self.HDU)

    @property
    def pipeline_FWHM(self):
        """
        The CCF FWHM as derived by the pipeline, from the header.
        """
        self._check_for_HDU()
        from .keywords import getFWHM
        return getFWHM(None, hdul=self.HDU)

    @property
    def pipeline_CONTRAST(self):
        """
        The CCF contrast as derived by the pipeline, from the header.
        """
        self._check_for_HDU()
        from .keywords import getCONTRAST
        return getCONTRAST(None, hdul=self.HDU)

    @property
    def pipeline_BIS(self):
        """
        The CCF BIS SPAN as derived by the pipeline, from the header.
        """
        self._check_for_HDU()
        from .keywords import getBIS
        return getBIS(None, hdul=self.HDU)

    @property
    def pipeline_RVerror(self):
        """  The RV error as derived by the pipeline, from the header."""
        self._check_for_HDU()
        return getRVerror(None, hdul=self.HDU)

    ############################################################################
    def recalculate_ccf(self, orders=None, weighted=False):
        """
        Recompute the final CCF from the CCFs of individual orders

        Args:
            orders (slice, int, tuple, list, array): 
                List of orders for which to sum the CCFs. If None, use all
                orders. If an int, return directly the CCF of that order
                (1-based).
        """
        if orders is None:
            return self.ccf

        warnings.warn('In this function, orders are 1-based. Make sure the right orders are being used!')

        if isinstance(orders, int):
            return self._SCIDATA[orders - 1]
        else:
            if isinstance(orders, slice):
                orders = np.arange(self.norders)[orders]
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
        Remove specific orders and recompute the CCF (CAREFUL: 1-based indexing)

        Args:
            orders (slice, int, tuple, list, array): 
                List of orders for which to sum the CCFs. If None, use all orders. 
                If an int, return directly the CCF of that order (1-based).
        """
        if isinstance(orders, int):
            orders = [orders]
        
        previous_RV = self.RV
        all_orders = np.arange(1, self.norders + 1)
        orders = np.setdiff1d(all_orders, orders)
        print('Recalculating CCF for subset of orders')
        self.ccf = self.recalculate_ccf(orders, weighted=weighted)
        if self.RV != previous_RV:
            print('RV changed from', previous_RV)
            print('             to', self.RV, 'km/s')


    def reset(self):
        """ Reset the CCF to the original CCF """
        self.ccf = self._SCIDATA[-1]


    def check(self, verbose=False):
        """ Check if the iCCF calculated values match the pipeline values """
        Q = {
            'RV': (self.RV, self.pipeline_RV),
            'RVerror': (self.RVerror, self.pipeline_RVerror),
            'FWHM': (self.FWHM, self.pipeline_FWHM),
            'contrast': (self.contrast, self.pipeline_CONTRAST),
            'BIS': (self.BIS, self.pipeline_BIS),
        }
        digits = self._precision
        matches = True
        for q, (val1, val2) in Q.items():
            if verbose:
                print(f'comparing {q:8s}: calculated/pipeline:', end=' ')
                mag1 = max(0, int(np.log10(np.abs(val1))))
                mag2 = max(0, int(np.log10(np.abs(val2))))
                print(f'{val1:.{digits - mag1}f} / {val2:.{digits - mag2}f}', end=' ')
            if np.allclose(val1, val2, rtol=0.0, atol=self._eps):
                if verbose:
                    print("✓")
            else:
                matches = False
                if verbose:
                    print(f"X (abs diff = {np.abs(val1 - val2)})")
        return matches

    @property
    def zero_ccfs(self):
        self._SCIDATA

    def to_dict(self):
        return writers.to_dict(self)

    def to_rdb(self, filename='stdout', clobber=False):
        return writers.to_rdb(self, filename, clobber)

    def plot(self, ax=None, show_fit=True, show_bisector=False,
             show_residuals=False, over=0):
        """ Plot the CCF, together with the Gaussian fit and the bisector """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)

        label = basename(self.filename)
        label = label.replace('%3A', ':')
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

        if show_residuals:
            ax.errorbar(self.rv, self.ccf - gauss(self.rv, pfit), eccf,
                        fmt='s-', ms=3, label='residuals')

        if show_bisector:
            out = BIS(self.rv, self.ccf, full_output=True)
            bisf = out[-1]
            top_limit = out[-4][1]
            bot_limit = out[-5][0]
            yy = np.linspace(bot_limit, top_limit, 25)
            ax.plot(bisf(yy), yy, '-s', ms=3,
                    label=f'bisector (BIS={self.BIS*1e3:.3f} m/s)')

        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.15))
        ax.set(xlabel='RV [km/s]', ylabel='CCF')
        return ax.figure, ax

    def _get_x_for_plot_individual_CCFs(self):
        n = self.individual_RV.size
        x = [
            (self.rv - self.rv[0]) / np.ptp(self.rv) - 0.5 + i
            for i in range(1, n + 1)
        ]
        return x

    def plot_individual_CCFs(self, ax=None, show_errors=True, show_fit=False, **kwargs):
        """ Plot the CCFs for individual orders """
        import matplotlib.pyplot as plt

        if ax is None:
            _, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 4),
                                        width_ratios=[4, 1.5],
                                        constrained_layout=True)
        else:
            ax2 = None

        n = self.individual_RV.size
        x = self._get_x_for_plot_individual_CCFs()

        show_errors = show_errors and self.eccf is not None

        for i in range(1, n + 1):
            if show_errors:
                ax.errorbar(x[i-1], self._SCIDATA[i - 1], self._ERRDATA[i - 1], **kwargs)
            else:
                ax.plot(x[i-1], self._SCIDATA[i - 1], **kwargs)

            if show_fit:
                try:
                    p = gaussfit(self.rv, self._SCIDATA[i - 1])
                    ax.plot(x[i-1], gauss(self.rv, p), 'k', alpha=0.7)
                    ax.text(i, np.max(self._SCIDATA[i - 1]), f'{p[1]:.3f} m/s',
                            rotation=60, fontsize=8)
                except RuntimeError:
                    pass

        ax.set(xlabel='spectral order', ylabel='CCF', xlim=(-5, n+5))

        if ax2:
            ax2.plot(self.rv, self.ccf, **kwargs)
            ax2.set(xlabel='RV [km/s]', ylabel='CCF')

        return ax.figure, ax

    def plot_individual_RV(self, ax=None, relative=False, **kwargs):
        """ Plot the RV for individual orders """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)

        n = self.individual_RV.size
        orders = np.arange(1, n + 1)

        kwargs.setdefault('fmt', 'o')

        if relative:
            ax.errorbar(orders, (self.individual_RV - self.RV) / self.individual_RVerror,
                        self.individual_RVerror, **kwargs, label='individual order RV')
            ax.axhline(0.0, color='darkgreen', ls='--', label='final RV')
        else:
            ax.errorbar(orders, self.individual_RV,
                        self.individual_RVerror, **kwargs, label='individual order RV')
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

        I = Indicators.from_file(f, **kwargs)  # noqa: E741
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


def add_order_by_order_info(Ind: Indicators):
    """
    Adds order-by-order RV and FWHM to the HDU of the indicators object `I`.
    """
    from astropy.io import fits
    Ind._warnings = False
    
    col1 = fits.Column(name='order', array=np.arange(1, Ind.norders + 1), format='I')

    # is SCIDATA a float32? (ignore endianness)
    if Ind._SCIDATA.dtype.str[1:] == np.float32().dtype.str[1:]:
        format = 'E'
    else:
        format = 'D'

    col2 = fits.Column(name='CCF RV', array=Ind.individual_RV, format=format)
    col3 = fits.Column(name='CCF RV ERROR', array=Ind.individual_RVerror, format=format)
    col4 = fits.Column(name='CCF FWHM', array=Ind.individual_FWHM, format=format)
    col5 = fits.Column(name='CCF FWHM ERROR', array=Ind.individual_FWHMerror, format=format)

    new_hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4, col5])
    new_hdu.name = 'ORDERRV'
    Ind.HDU.append(new_hdu)

    return Ind


def recreate_ccf_file(file, output=None, replace_rv=True, check=False,
                      overwrite=False):
    Ind = Indicators.from_file(file, keep_open=True)
    assert isinstance(Ind, Indicators)
    
    new_file = output or file.replace('.fits', '.iccf.fits')
    if os.path.exists(new_file) and not overwrite:
        print(f'output file ({new_file}) already exists, not overwriting')
        return

    if check:
        Ind.check(verbose=True)

    if replace_rv:
        Ind.HDU[0].header['HIERARCH ESO QC CCF RV'] = Ind.RV
        Ind.HDU[0].header['HIERARCH ESO QC CCF RV ERROR'] = Ind.RVerror
        Ind.HDU[0].header['HIERARCH ESO QC CCF FWHM'] = Ind.FWHM
        Ind.HDU[0].header['HIERARCH ESO QC CCF FWHM ERROR'] = Ind.FWHMerror
        Ind.HDU[0].header['HIERARCH ESO QC CCF CONTRAST'] = Ind.contrast
        Ind.HDU[0].header['HIERARCH ESO QC CCF CONTRAST ERROR'] = Ind.contrast_error

    print(f'Adding order-by-order RV and FWHM to {file}')
    Ind = add_order_by_order_info(Ind)

    # order_by_order_values = (
    #     I.individual_RV, I.individual_RVerror,
    #     I.individual_FWHM, I.individual_FWHMerror
    # )
    # for i, (rv, erv, fwhm, efwhm) in enumerate(zip(*order_by_order_values)):
    #     if np.isnan(rv) or np.isnan(erv):
    #         rv, erv = 'NaN', 'NaN'
    #         fwhm, efwhm = 'NaN', 'NaN'

    #     for k, v in (
    #         (f'HIERARCH ICCF ORDER{i+1} CCF RV',         (rv,    f'Radial velocity of order {i+1} [km/s]')                if add_comments else rv   ),
    #         (f'HIERARCH ICCF ORDER{i+1} CCF RV ERROR',   (erv,   f'Uncertainty on radial velocity of order {i+1} [km/s]') if add_comments else erv  ),
    #         (f'HIERARCH ICCF ORDER{i+1} CCF FWHM',       (fwhm,  f'CCF FWHM of order {i+1} [km/s]')                       if add_comments else fwhm ),
    #         (f'HIERARCH ICCF ORDER{i+1} CCF FWHM ERROR', (efwhm, f'Uncertainty on CCF FWHM of order {i+1} [km/s]')        if add_comments else efwhm),
    #     ):
    #         I.HDU[0].header[k] = v

    Ind.HDU.writeto(new_file, output_verify='exception', checksum=True,
                  overwrite=overwrite)
    Ind.HDU.close()

    print('Wrote', new_file)
    return new_file


def bjd(ind: list):
    return np.array([i.bjd for i in ind])

def vrad(ind: list):
    return np.array([i.RV for i in ind])

def svrad(ind: list):
    return np.array([i.RVerror for i in ind])