""" Calculate CCFs themselves. So meta! """

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from bisect import bisect_left, bisect_right

from .utils import doppler_shift_wave


# -----------------------------------------------------------------------------
# crosscorrRV calculates the CCF by **shifting the observed spectrum**
# This is very likely **not** the best way to do it.
# (in addition, the function also ignores the width of the mask)
# Consider using the makeCCF function instead.
# -----------------------------------------------------------------------------
# # based on PyAstronomy.pyasl.crosscorrRV
# https://github.com/sczesla/PyAstronomy/blob/master/src/pyasl/asl/crosscorr.py
def crosscorrRV(wave, flux, mask_wave, mask_contrast, rvmin=None, rvmax=None,
                drv=None, rvarray=None):
    """
    Cross-correlate a spectrum with a mask template.
    
    For each RV shift in rvmin:rvmax:drv (or in rvarray), the wavelength axis 
    of the spectrum is Doppler shifted. Then, the shifted spectrum is linearly 
    interpolated at the wavelength points of the template (mask) to calculate 
    the cross-correlation function.
    
    Parameters
    ----------
    wave : array
        The wavelength of the observed spectrum.
    flux : array
        The flux of the observed spectrum.
    mask_wave : array
        The wavelength of the mask.
    mask_contrast : array
        The flux (contrast) of the mask.
    rvmin : float
        Minimum radial velocity for which to calculate the CCF [km/s].
    rvmax : float
        Maximum radial velocity for which to calculate the CCF [km/s].
    drv : float
        The radial-velocity step [km/s].
    rvarray : array
        The radial velocities at which to calculate the CCF. If `rvarray` is
        provided, `rvmin`, `rvmax` and `drv` are ignored.
    
    Returns
    -------
    rv : array
        The radial-velocity where the CCF was calculated [km/s]. These RVs 
        refer to a shift of the template -- positive values indicate that the 
        template has been red-shifted and negative numbers indicate a 
        blue-shift of the template.
    ccf : array
        The values of the cross-correlation function.
    """

    # Speed of light in km/s
    c = 299792.458

    if rvarray is None:
        if rvmin is None or rvmax is None or drv is None:
            raise ValueError("Provide `rvmin`, `rvmax`, and `drv`.")
        # Check order of rvmin and rvmax
        if rvmax <= rvmin:
            raise ValueError("`rvmin` should be smaller than `rvmax`.")
        # We shift the spectrum, not the mask, but to be consistent with what
        # the user might expect, define the rv array "backwards"
        drvs = np.arange(rvmax, rvmin - drv / 2, -drv)
    else:
        drvs = rvarray

    # Calculate the cross correlation
    cc = np.zeros(len(drvs))
    for i, rv in enumerate(drvs):
        # Apply the Doppler shift (to the observed spectrum)
        # with bounds_error=False and fill_value=0, interp1d will extrapolate
        # to 0 outside of the input wavelength
        fi = interp1d(wave * (1.0 + rv / c), flux, bounds_error=False,
                      fill_value=0)

        # Shifted spectrum evaluated at locations of template mask
        cc[i] = np.sum(mask_contrast * fi(mask_wave))

    return drvs, cc


def makeCCF(spec_wave, spec_flux, mask_wave=None, mask_contrast=None, mask=None,
            mask_width=0.82, rvmin=None, rvmax=None, drv=None, rvarray=None):
    """
    Cross-correlate an observed spectrum with a mask template.

    For each RV value in rvmin:rvmax:drv (or in rvarray), the wavelength axis of
    the mask is Doppler shifted. The mask is then projected onto the spectrum
    (using the provided `mask_width`) and the sum of the flux that goes through
    the mask "holes" is calculated. This sum is weighted by the mask contrast
    (which corresponds to line depths) to optimally extract the Doppler
    information.

    Parameters
    ----------
    spec_wave : array The wavelength of the observed spectrum. spec_flux : array
        The flux of the observed spectrum. mask_wave : array, optional The
        wavelength of the mask. mask_contrast : array, optional The flux
        (contrast) of the mask. mask : array (..., 3), optional The mask as an
        array with lambda1, lambda2, depth rvmin : float, optional Minimum
        radial velocity for which to calculate the CCF [km/s]. rvmax : float,
        optional Maximum radial velocity for which to calculate the CCF [km/s].
        drv : float, optional The radial-velocity step [km/s]. rvarray : array,
        optional The radial velocities at which to calculate the CCF. If this is
        provided, `rvmin`, `rvmax` and `drv` are ignored.

    Returns
    -------
    rv : array The radial-velocity where the CCF was calculated [km/s]. These
        RVs refer to a shift of the template -- positive values indicate that
        the template has been red-shifted and negative numbers indicate a
        blue-shift of the template. ccf : array The values of the
        cross-correlation function.
    """
    if rvarray is None:
        if rvmin is None or rvmax is None or drv is None:
            raise ValueError("Provide `rvmin`, `rvmax`, and `drv`.")
        # Check order of rvmin and rvmax
        if rvmax <= rvmin:
            raise ValueError("`rvmin` should be smaller than `rvmax`.")
        rvarray = np.arange(rvmin, rvmax + drv / 2, drv)

    wave_resolution = spec_wave[1] - spec_wave[0]

    if mask is None:
        mask = np.c_[doppler_shift_wave(mask_wave, -mask_width / 2),
                    doppler_shift_wave(mask_wave, mask_width / 2), mask_contrast]

    ccfarray = np.zeros_like(rvarray)
    for i, RV in enumerate(rvarray):
        nlines = 0
        CCF = 0.0

        mask_rv_shifted = np.copy(mask)
        mask_rv_shifted[:, :2] = doppler_shift_wave(mask[:, :2], RV)

        region = (spec_wave[0] < mask_rv_shifted[:, 0]) & (
            mask_rv_shifted[:, 1] < spec_wave[-1])
        mask_rv_shifted = mask_rv_shifted[region]

        for mask_line_start, mask_line_end, mask_line_depth in mask_rv_shifted:
            if mask_line_end + wave_resolution >= spec_wave[-1]:
                break

            linePixelIni = bisect_left(spec_wave, mask_line_start)
            linePixelEnd = bisect_right(spec_wave, mask_line_end)

            lineFractionIni = (
                spec_wave[linePixelIni] - mask_line_start) / wave_resolution
            lineFractionEnd = (
                mask_line_end - spec_wave[linePixelEnd]) / wave_resolution

            CCF += mask_line_depth * (
                np.sum(spec_flux[linePixelIni:linePixelEnd]) +
                lineFractionIni * spec_flux[linePixelIni - 1] +
                lineFractionEnd * spec_flux[linePixelEnd + 1])
            nlines += 1

        ccfarray[i] = CCF

    return ccfarray
