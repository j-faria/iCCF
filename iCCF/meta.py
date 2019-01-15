""" Calculate CCFs themselves. So meta! """

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# based on PyAstronomy.pyasl.crosscorrRV
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