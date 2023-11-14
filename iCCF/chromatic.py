import os
import bisect
import warnings
from astropy.timeseries import periodograms
try:
    from pkg_resources import resource_stream
    pkg_resources_available = True
except ModuleNotFoundError:
    pkg_resources_available = False


import numpy as np
from numpy import sqrt, sum
import matplotlib.pyplot as plt

from astropy.io import fits
from cached_property import cached_property

from .iCCF import Indicators
from .gaussian import gaussfit, RV, RVerror, FWHM, FWHMerror
from .keywords import getRV, getRVarray
from .utils import find_myself
# from .utils import get_orders_mean_wavelength


def read_spectral_format():
    if pkg_resources_available:
        sf_red_stream = resource_stream(__name__, 'data/spectral_format_red.dat')
        sf_blue_stream = resource_stream(__name__, 'data/spectral_format_blue.dat')
    else:
        sf_red_stream = os.path.join(os.path.dirname(__file__), 'data/spectral_format_red.dat')
        sf_blue_stream = os.path.join(os.path.dirname(__file__), 'data/spectral_format_blue.dat')

    red = np.loadtxt(sf_red_stream)
    blue = np.loadtxt(sf_blue_stream)

    col_start_wave = 7
    col_end_wave = 8

    order_wave_range = {}

    for i, order in enumerate(blue[::-1]):
        order_range = [order[col_start_wave], order[col_end_wave]]
        order_wave_range[i] = order_range

    for i, order in enumerate(red[::-1], start=i+1):
        order_range = [order[col_start_wave], order[col_end_wave]]
        order_wave_range[i] = order_range

    return order_wave_range


class chromaticRV():
    def __init__(self, indicators):
        """
        indicators : Indicators or list
            Instance or list of instances of iCCF.Indicators
        """
        self.order_wave_range = read_spectral_format()
        self.wave_starts = [v[0] for v in self.order_wave_range.values()]
        self.wave_ends = [v[1] for v in self.order_wave_range.values()]

        self._blue_wave_limits = (440, 570)
        self._mid_wave_limits = (570, 690)
        self._red_wave_limits = (730, 790)

        self._slice_policy = 0  # by default use both slices

        self.blue_orders = self._find_orders(self._blue_wave_limits)
        self.mid_orders = self._find_orders(self._mid_wave_limits)
        self.red_orders = self._find_orders(self._red_wave_limits)

        self._blueRV = None
        self._midRV = None
        self._redRV = None

        self._blueRVerror = None
        self._midRVerror = None
        self._redRVerror = None

        self.n = len(indicators)
        if self.n == 1:
            indicators = [indicators, ]
        self.I = self.indicators = indicators
        # store all but the last CCF for each of the Indicators instances
        self.ccfs = [i.HDU[i._hdu_number].data[:-1] for i in self.I]
        # store the last CCFs separately
        self.ccf = [i.HDU[i._hdu_number].data[-1] for i in self.I]
        # try storing the CCF uncertainties as well
        self.eccfs = []
        for i in self.I:
            try:
                self.eccfs.append(i.HDU[2].data[:-1])
            except IndexError:
                self.eccfs.append(None)


    def __repr__(self):
        bands = ', '.join(map(repr, self.bands))
        nb = len(self.bands)
        return f'chromaticRV({self.n} CCFs; {nb} bands: {bands} nm)'

    @property
    def blue_wave_limits(self):
        """ Wavelength limits for the blue RV calculations [nm] """
        return self._blue_wave_limits

    @blue_wave_limits.setter
    def blue_wave_limits(self, vals):
        assert len(vals) == 2, 'provide two wavelengths (start and end) in nm'
        self.blue_orders = self._find_orders(vals)
        self._blue_wave_limits = vals
        self._blueRV, self._midRV, self._redRV = None, None, None

    @property
    def mid_wave_limits(self):
        """ Wavelength limits for the mid RV calculations [nm] """
        return self._mid_wave_limits

    @mid_wave_limits.setter
    def mid_wave_limits(self, vals):
        assert len(vals) == 2, 'provide two wavelengths (start and end) in nm'
        self.mid_orders = self._find_orders(vals)
        self._mid_wave_limits = vals
        self._blueRV, self._midRV, self._redRV = None, None, None

    @property
    def red_wave_limits(self):
        """ Wavelength limits for the red RV calculations [nm] """
        return self._red_wave_limits

    @red_wave_limits.setter
    def red_wave_limits(self, vals):
        assert len(vals) == 2, 'provide two wavelengths (start and end) in nm'
        self.red_orders = self._find_orders(vals)
        self._red_wave_limits = vals
        self._blueRV, self._midRV, self._redRV = None, None, None

    @property
    def slice_policy(self):
        """ How to deal with the two order slices.
        0: use both slices by adding the corresponding CCFs (default)
        1: use only the first slice
        2: use only the second slice
        """
        return self._slice_policy

    @slice_policy.setter
    def slice_policy(self, val):
        self._slice_policy = val
        self.blue_orders = self._find_orders(self._blue_wave_limits)
        self.mid_orders = self._find_orders(self._mid_wave_limits)
        self.red_orders = self._find_orders(self._red_wave_limits)
        self._blueRV, self._midRV, self._redRV = None, None, None


    def _find_orders(self, wave_limits):
        order_start = bisect.bisect(self.wave_starts, wave_limits[0])
        order_end = bisect.bisect(self.wave_ends, wave_limits[1])

        order_start = order_start * 2
        order_end = order_end * 2 + 1

        if self.slice_policy == 0: # using both order slices
            step = 1
            return slice(order_start, order_end+1, step)

        elif self.slice_policy == 1:  # using first slice only
            step = 2
            return slice(order_start, order_end+1, step)

        elif self.slice_policy == 2:  # using second slice only
            step = 2
            return slice(order_start+1, order_end+1, step)


    def get_rv(self, orders):
        """ Get radial velocity, FWHM and uncertainties for specific orders

        orders : int, slice, tuple, array
            The CCFs of these orders will be summed to calculate the RV.
            If int, only the CCF at that index will be used.
            If slice, orders in the slice's range will be used.
            If tuple, should have length 2 or 3 and correspond to minimum index,
            maximum index and possibly step, for which orders to use
            If array, should contain indices of orders to use
        """
        if isinstance(orders, int):
            orders = slice(orders, orders + 1)
        elif isinstance(orders, tuple):
            orders = slice(*orders)

        rv, rve = [], []
        fwhm, fwhme = [], []

        for i, full_ccf, full_eccf in zip(self.I, self.ccfs, self.eccfs):
            # create the CCF
            ccf = full_ccf[orders].sum(axis=0)
            if full_eccf is not None:
                eccf = np.sqrt(np.square(full_eccf[orders]).sum(axis=0))
            else:
                eccf = None

            # calculate RV and RV error
            rv.append(RV(i.rv, ccf, eccf))
            rve.append(RVerror(i.rv, ccf, eccf))
            # rve.append(np.nan)

            # calculate FWHM and FWHM error
            fwhm.append(FWHM(i.rv, ccf))
            fwhme.append(FWHMerror(i.rv, ccf, eccf))

        # if not has_errors:
        #     warnings.warn(
        #         'Cannot access CCF uncertainties to calculate RV error')
        #     return np.array(rv), None
        # else:
        return map(np.array, (rv, rve, fwhm, fwhme))


    @property
    def bands(self):
        """ Wavelength limits of blue, mid, and red bands """
        b = self.blue_wave_limits, self.mid_wave_limits, self.red_wave_limits
        return b

    @cached_property
    def time(self):
        """ BJD of observations """
        return np.fromiter((i.bjd for i in self.I), np.float, self.n)

    @property
    def blueRV(self):
        if self._blueRV is None:
            out = self.get_rv(self.blue_orders)
            self._blueRV, self._blueRVerror, self._blueFWHM, self._blueFWHMerror = out
        return self._blueRV

    @property
    def midRV(self):
        if self._midRV is None:
            out = self.get_rv(self.mid_orders)
            self._midRV, self._midRVerror, self._midFWHM, self._midFWHMerror = out
        return self._midRV

    @property
    def redRV(self):
        if self._redRV is None:
            out = self.get_rv(self.red_orders)
            self._redRV, self._redRVerror, self._redFWHM, self._redFWHMerror = out
        return self._redRV

    @property
    def fullRV(self):
        return np.fromiter((i.RV for i in self.I), np.float, self.n)

    @property
    def fullRVerror(self):
        return np.fromiter((i.RVerror for i in self.I), np.float, self.n)


    def bin(self, night_indices):
        u = np.unique(night_indices)

        ccfs = np.array(self.ccfs)  # shape: (Nobs, Norders, Nrv)
        ccfsb = [ccfs[night_indices == i].mean(axis=0) for i in u]
        ccfsb = np.array(ccfsb)  # shape: (Nobs_binned, Norders, Nrv)
        self.ccfs = ccfsb

        eccfs = np.array(self.eccfs)  # shape: (Nobs, Norders, Nrv)
        eccfsb = [sqrt(sum(eccfs[night_indices == i]**2, axis=0)) for i in u]
        eccfsb = np.array(eccfsb)  # shape: (Nobs_binned, Norders, Nrv)
        self.eccfs = eccfsb

        ccf = np.array(self.ccf)  # shape: (Nobs, Nrv)
        ccfb = [ccf[night_indices == i].mean(axis=0) for i in u]
        ccfb = np.array(ccfb)  # shape: (Nobs_binned, Nrv)
        self.ccf = ccfb

        rv = self.I[0].rv
        self.indicators = [Indicators(rv, ccf.sum(axis=0)) for ccf in ccfsb]
        self.I = self.indicators
        self.n = len(self.I)

    def plot(self, periodogram=False, mask=None, obs=None):
        ncols = 2 if periodogram else 1

        fig, axs = plt.subplots(3 + 1, ncols, constrained_layout=True)
        axs = axs.ravel()

        if periodogram:
            indices_plots = np.arange(0, 8, 2)
            indices_pers = np.arange(1, 8, 2)
            for ax in axs[indices_pers[1:]]:
                ax.sharex(axs[indices_pers[0]])
                ax.sharey(axs[indices_pers[0]])
        else:
            indices_plots = np.arange(0, 4)

        for ax in axs[indices_plots[1:]]:
            ax.sharex(axs[indices_plots[0]])
            ax.sharey(axs[indices_plots[0]])

        kw = dict(fmt='o', ms=2)

        if mask is None:
            mask = np.ones_like(self.time, dtype=bool)

        axs[indices_plots[0]].errorbar(self.time[mask],
                                       1e3*(self.fullRV[mask] - self.fullRV[mask].mean()),
                                       self.fullRVerror[mask], color='k', **kw)

        axs[indices_plots[1]].errorbar(self.time[mask],
                                       1e3*(self.blueRV[mask] - self.blueRV[mask].mean()),
                                       self._blueRVerror[mask], color='b', **kw)
        axs[indices_plots[2]].errorbar(self.time[mask],
                                       1e3*(self.midRV[mask] - self.midRV[mask].mean()),
                                       self._midRVerror[mask], color='g', **kw)
        axs[indices_plots[3]].errorbar(self.time[mask],
                                       1e3*(self.redRV[mask] - self.redRV[mask].mean()),
                                       self._redRVerror[mask], color='r', **kw)

        if periodogram:
            periods = np.logspace(np.log10(1), np.log10(2 * self.time.ptp()),
                                  1000)
            kwfap = dict(alpha=0.2, ls='--')

            if obs is None:
                from astropy.timeseries import LombScargle
                def gls(t, y, e, *args):
                    model = LombScargle(t, y, e)
                    return model, model.power(1 / periods)
            else:
                from gatspy import periodic
                def gls(t, y, e, obs):
                    model = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=0)
                    model.fit(t, y, e, filts=obs)
                    power = model.periodogram(periods)
                    model.false_alarm_level = lambda x: np.zeros_like(x)
                    return model, power

            models = []

            model, power = gls(self.time[mask], self.fullRV[mask], self.fullRVerror[mask], obs[mask])
            models.append(model)
            axs[1].semilogx(periods, power, color='k')
            if hasattr(model, 'false_alarm_level'):
                axs[1].hlines(model.false_alarm_level([0.1, 0.01]),
                              *axs[1].get_xlim(), **kwfap)
            if obs is not None:
                axs[indices_plots[0]].plot(self.time[mask], 1e3*(model.ymean_ - self.fullRV[mask].mean()), ls='--')

            model, power = gls(self.time[mask], self.blueRV[mask], self._blueRVerror[mask], obs[mask])
            models.append(model)
            axs[3].semilogx(periods, power, color='b')
            if hasattr(model, 'false_alarm_level'):
                axs[3].hlines(model.false_alarm_level([0.1, 0.01]),
                              *axs[1].get_xlim(), **kwfap)
            if obs is not None:
                axs[indices_plots[1]].plot(self.time[mask], 1e3*(model.ymean_ - self.blueRV[mask].mean()), ls='--')

            model, power = gls(self.time[mask], self.midRV[mask], self._midRVerror[mask], obs[mask])
            models.append(model)
            axs[5].semilogx(periods, power, color='g')
            if hasattr(model, 'false_alarm_level'):
                axs[5].hlines(model.false_alarm_level([0.1, 0.01]),
                              *axs[1].get_xlim(), **kwfap)
            if obs is not None:
                axs[indices_plots[2]].plot(self.time[mask], 1e3*(model.ymean_ - self.midRV[mask].mean()), ls='--')

            model, power = gls(self.time[mask], self.redRV[mask], self._redRVerror[mask], obs[mask])
            models.append(model)
            axs[7].semilogx(periods, power, color='r')
            if hasattr(model, 'false_alarm_level'):
                axs[7].hlines(model.false_alarm_level([0.1, 0.01]),
                              *axs[1].get_xlim(), **kwfap)
            if obs is not None:
                axs[indices_plots[3]].plot(self.time[mask], 1e3*(model.ymean_ - self.redRV[mask].mean()), ls='--')

        for ax in axs[indices_plots]:
            ax.set_ylabel('RV [m/s]')

        axs[indices_plots[-1]].set_xlabel('Time [BJD]')

        if periodogram:
            axs[indices_pers[0]].set_xlim((periods.min(), periods.max()))
            axs[indices_pers[-1]].set_xlabel('Period [days]')

            kw = dict(fontsize=8)
            axs[indices_pers[0]].set_title('full $\lambda$ range', loc='right', **kw)
            axs[indices_pers[1]].set_title(f'blue $\lambda={self.bands[0]}$ nm', loc='right', **kw)
            axs[indices_pers[2]].set_title(f'mid $\lambda={self.bands[1]}$ nm', loc='right', **kw)
            axs[indices_pers[3]].set_title(f'red $\lambda={self.bands[2]}$ nm', loc='right', **kw)

        for ax in axs[indices_pers]:
            ax.axvline(5.12, alpha=0.2, color='k', ls='--', zorder=-1)
            ax.axvline(11.19, alpha=0.2, color='k', ls='--', zorder=-1)
        axs[indices_pers[0]].set_xlim(0.9, 200)

        return fig, axs, models


    def plot_ccfs(self, orders=None, show_filenames=False):
        if orders is None:
            orders = slice(None, None)
        elif isinstance(orders, int):
            orders = slice(orders, orders + 1)
        elif isinstance(orders, tuple):
            orders = slice(*orders)

        fig, ax = plt.subplots(1, 1)  #, constrained_layout=True)
        for i in self.I:
            line = ax.plot(i.rv, i._SCIDATA[orders].T)
            if show_filenames:
                color = line[0].get_color()
                ax.text(i.rv[0], i.ccf[0], i.filename, fontsize=8, color=color)

        ax.set(xlabel='RV', ylabel='CCF')




def each_order_rv(rv, ccfs, exclude_last=True):
    """
    Calculate RVs for each spectral order by fitting Gaussians to individual CCFs

    Parameters
    ----------
    rv : array
        Radial velocities where each CCF is defined
    ccfs : array
        The CCFs for each spectral order (order o, radial velocity rv)
    exclude_last : bool
        Whether to exclude the last index of ccfs (usually the sum of all other 
        CCFs) from the calculation.
    
    Returns
    -------
    rvs : array
        The center of a Gaussian fit to each order's CCF
    """
    last = -1 if exclude_last else None

    gen = (gaussfit(rv, ccf)[1] for ccf in ccfs[:last])
    rvs = np.fromiter(gen, dtype=float)

    return rvs


def rv_color(rv, ccfs, blue=slice(0,80), red=slice(80,-1), avoid_blue=0, gap=0):
    """
    Calculate the RV color by combining blue and red CCFs

    Parameters
    ----------
    rv : array
        Radial velocities where each CCF is defined
    ccfs : array
        The CCFs for each spectral order (order o, radial velocity rv)
    blue : slice
        A slice object with the start and stop indices of the blue orders. The
        default (0:80) is for ESPRESSO. For HARPS, use ...
    red : slice
        A slice object with the start and stop indices of the red orders. The
        default (80:-1) is for ESPRESSO. For HARPS, use ...
    avoid_blue : int
        How many orders to skip in the bluest part of the spectrum. This will 
        be added to the beginning of the `blue` slice
    gap : int or tuple
        If an integer, the number of orders to remove from the "middle" for both
        blue and red parts. If a tuple, the number of orders to remove from the 
        blue and red, respectively
    """
    if isinstance(gap, tuple):
        gap_blue, gap_red = gap
    elif isinstance(gap, int):
        gap_blue = gap_red = gap
    else:
        raise ValueError(f"`gap` should be int or tuple, got {gap}")

    blue = slice(blue.start + avoid_blue, blue.stop - gap_blue)
    red = slice(red.start + gap_red, red.stop)

    ccf_blue = ccfs[blue, :].sum(axis=0)
    ccf_red = ccfs[red, :].sum(axis=0)

    rv_blue = gaussfit(rv, ccf_blue)[1]
    rv_red = gaussfit(rv, ccf_red)[1]
    print(rv_blue, rv_red)



# def chromatic_index(rv, ccfs, wave, rvpipe=None):
#     """ 
#     Calculate the chromatic index, as described in Zechmeister et al. (2018).

#     Parameters
#     ----------
#     rv : array
#         Radial velocities where each CCF is defined   
#     """
#     if isinstance(wave, str): # assume it's a filename
#         wave = get_wave(wave)
#     elif isinstance(wave, np.ndarray):
#         pass
#     else:
#         raise ValueError('`wave` should be filename or array with wavelengths')

#     mean_wave = get_orders_mean_wavelength(wave, log=True)
#     rvs = each_order_rv(rv, ccfs)

#     ind = ~np.isnan(rvs)
#     p = np.polyfit(np.log(mean_wave[ind]), rvs[ind], 1)

#     if rvpipe is None:
#         rvpipe = gaussfit(rv, ccfs[-1])[1]

#     beta = p[0]
#     lv = np.exp(abs((p[1] - rvpipe)/p[0]))
#     return beta, lv



# def chromatic_index_from_files(s2dfile, ccffile):
#     """ 
#     Calculate the chromatic index, as described in Zechmeister et al. (2018).

#     Parameters
#     ----------
#     s2dfile : str
#         Filename of the S2D fits file
#     ccffile : str
#         Filename of the CCF fits file
#     """
#     wave = get_wave(s2dfile)
#     mean_wave = get_orders_mean_wavelength(wave, log=True)

#     rvpipe = getRV(ccffile)
#     rv = getRVarray(ccffile)

#     ccfs = fits.open(ccffile)[1].data
#     rvs = each_order_rv(rv, ccfs)

#     return chromatic_index(rv, ccfs, wave, rvpipe)
