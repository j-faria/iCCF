from typing import List, Optional

import numpy as np
from numpy import exp, log, sqrt
from scipy import optimize
from scipy.interpolate import CubicSpline

from .utils import numerical_gradient

def gauss(x, p):
    """ A Gaussian function with parameters p = [A, x0, σ, offset]. """
    return p[0] * exp(-(x - p[1])**2 / (2 * p[2]**2)) + p[3]


def _gauss_partial_deriv(x, p):
    """ Partial derivatives of a Gaussian with respect to each parameter. """
    A, x0, sig, _ = p
    g = gauss(x, [A, x0, sig, 0.0])
    dgdA = gauss(x, [1.0, x0, sig, 0.0])
    dgdx0 = g * ((x - x0) / sig**2)
    dgdsig = g * ((x - x0)**2 / sig**3)
    dgdoffset = np.ones_like(x)
    return np.c_[dgdA, dgdx0, dgdsig, dgdoffset]


def _gauss_initial_guess(x, y):
    """ Educated guess (from the data) for Gaussian parameters. """
    # these guesses tend to work better for narrow-ish gaussians
    p0 = []

    ## guess the amplitude 
    ## range of CCF values (force it to be negative)
    p0.append(-abs(np.ptp(y)))

    ## guess the center, but maybe the CCF is upside down?
    # m = y.mean()
    # ups_down = np.sign(np.percentile(y, 50) - m) != np.sign(y.max() - m)
    # if ups_down:  # seems like it
    #     warnings.warn('It seems the CCF might be upside-down?')
    #     p0[0] *= -1
    #     p0.append(x[y.argmax()])
    # else:
    p0.append(x[y.argmin()])
    
    ## guess the width
    ## assume at least 5 points "in" the CCF and estimate the RV span
    if x.size > 5:
        p0.append(np.ptp(x[y < np.sort(y)[5]]))
    else:
        ## sigma always fixed to 1 works most times
        p0.append(1) 
    ## another option, from arXiv:1907.07241, but sometimes it's too small
    # p0.append( abs(_ccf_trapz(x, y)) / np.ptp(x) / np.sqrt(2*np.pi) )

    # guess the offset
    # mean of first and last CCF values
    p0.append(0.5 * (y[0] + y[-1]))
    
    return p0


def _gauss_initial_guess_pipeline(x, y):
    """ Initial values for Gaussian parameters, as implemented in the ESPRESSO pipepline """
    c = np.max(y)
    fwhm = np.ptp(x) / 3
    x0 = x[y.argmin()]
    # ind[j] = (x0[j] - data->x[0])/(data->x[n-1] - data->x[0])*n;
    # not sure if int conversion is the same as in C
    ind = int((x0-x[0]) / (np.ptp(x)) * x.size)
    k = y[ind] - c
    return [k, x0, fwhm2sig(fwhm), c]



def gaussfit(x: np.ndarray,
             y: np.ndarray,
             p0: Optional[List] = None,
             yerr: Optional[np.ndarray] = None,
             return_errors: bool = False,
             use_deriv: bool = True,
             guess_rv: Optional[float] = None,
             **kwargs) -> List:
    """
    Fit a Gaussian function to `x`,`y` (and, if provided, `yerr`) using
    least-squares, with initial guess `p0` = [A, x0, σ, offset]. If p0 is not
    provided, the function tries an educated guess, which might lead to bad
    results.

    Args:
        x (array):
            The independent variable where the data is measured
        y (array):
            The dependent data.
        p0 (list or array):
            Initial guess for the parameters. If None, try to guess them from x,y.
        return_errors (bool):
            Whether to return estimated errors on the parameters.
        use_deriv (bool):
            Whether to use partial derivatives of the Gaussian (wrt the parameters)
            as Jacobian in the fit. If False, the Jacobian will be estimated.
        guess_rv (float):
            Initial guess for the RV (x0)
        **kwargs:
            Keyword arguments passed to `scipy.optimize.curve_fit`

    Returns:
        p (array):
            Best-fit values of the four parameters [A, x0, σ, offset]
        err (array):
            Estimated uncertainties on the four parameters
            (only if `return_errors=True`)
    """
    if (y == 0).all():
        return np.full(4, np.nan)

    x = x.astype(y.dtype.type)
    y = y.astype(y.dtype.type)

    if p0 is None:
        p0 = _gauss_initial_guess(x, y)
    if guess_rv is not None:
        p0[1] = guess_rv

    p0 = np.array(p0).astype(y.dtype.type)

    f = lambda x, *p: gauss(x, p)
    if use_deriv:
        df = lambda x, *p: _gauss_partial_deriv(x, p)
    else:
        df = None

    # bounds = ([-np.inf]*4, [np.inf]*4)
    # bounds[0][1] = x.min()
    # bounds[1][1] = x.max()
    # print(p0)
    # print(bounds)

    pfit, pcov, *_ = optimize.curve_fit(f, x, y, p0=p0, sigma=yerr, jac=df,
                                        xtol=1e-12, ftol=1e-14, check_finite=True, **kwargs)

    if 'full_output' in kwargs:
            infodict = _

    pfit = pfit.astype(y.dtype.type)

    if return_errors:
        errors = np.sqrt(np.diag(pcov))
        if 'full_output' in kwargs:
            return pfit, errors, infodict
        return pfit, errors

    if 'full_output' in kwargs:
        return pfit, infodict

    return pfit


def gaussfit_slope(x: np.ndarray,
                   y: np.ndarray,
                   p0: Optional[List] = None,
                   yerr: Optional[np.ndarray] = None,
                   return_errors: bool = False,
                   use_deriv: bool = True,
                   guess_rv: Optional[float] = None,
                   **kwargs) -> List:
    """
    Fit a Gaussian function and a slope to `x`,`y` (and, if provided, `yerr`)
    using least-squares, with initial guess `p0` = [A, x0, σ, offset, slope]. If
    p0 is not provided, the function tries an educated guess, which might lead
    to bad results.

    Args:
        x (array):
            The independent variable where the data is measured
        y (array):
            The dependent data.
        p0 (list or array):
            Initial guess for the parameters. If None, try to guess them from x,y.
        return_errors (bool):
            Whether to return estimated errors on the parameters.
        use_deriv (bool):
            Whether to use partial derivatives of the Gaussian (wrt the parameters)
            as Jacobian in the fit. If False, the Jacobian will be estimated.
        guess_rv (float):
            Initial guess for the RV (x0)
        **kwargs:
            Keyword arguments passed to `scipy.optimize.curve_fit`

    Returns:
        p (array):
            Best-fit values of the four parameters [A, x0, σ, offset, slope]
        err (array):
            Estimated uncertainties on the four parameters
            (only if `return_errors=True`)
    """
    if (y == 0).all():
        return np.full(4, np.nan)

    x = x.astype(y.dtype.type)
    y = y.astype(y.dtype.type)

    if p0 is None:
        p0 = _gauss_initial_guess(x, y)
    if guess_rv is not None:
        p0[1] = guess_rv

    p0.append(0.0)
    p0 = np.array(p0).astype(y.dtype.type)

    def f(x, *p):
        return gauss(x, p[:-1]) + p[-1] * (x - p[1])

    if use_deriv:
        raise NotImplementedError

    df = None

    pfit, pcov, *_ = optimize.curve_fit(f, x, y, p0=p0, sigma=yerr, jac=df,
                                        xtol=1e-12, ftol=1e-14, check_finite=True, **kwargs)

    if 'full_output' in kwargs:
            infodict = _

    pfit = pfit.astype(y.dtype.type)

    if return_errors:
        errors = np.sqrt(np.diag(pcov))
        if 'full_output' in kwargs:
            return pfit, errors, infodict
        return pfit, errors

    if 'full_output' in kwargs:
        return pfit, infodict

    return pfit



def sig2fwhm(sig):
    """ Convert standard deviation to full width at half maximum. """
    c = (2 * sqrt(2 * log(2))).astype(sig.dtype)
    return c * sig


def fwhm2sig(fwhm):
    """ Convert full width at half maximum to standard deviation. """
    c = (2 * sqrt(2 * log(2))).astype(fwhm.dtype)
    return fwhm / c


def RV(rv, ccf, eccf=None, **kwargs):
    """
    Calculate the radial velocity as the center of a Gaussian fit the CCF.
    
    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.
    kwargs : dict
        Keyword arguments passed directly to gaussfit
    """
    _, rv, _, _ = gaussfit(rv, ccf, yerr=eccf, **kwargs)
    return rv


def RVerror(rv, ccf, eccf):
    """
    Calculate the uncertainty on the radial velocity, following the same steps
    as the ESPRESSO DRS pipeline.

    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.
    eccf : array
        The errors on each value of the CCF profile.
    """
    ccf_slope = numerical_gradient(rv, ccf)
    ccf_sum = np.sum((ccf_slope / eccf)**2)
    return 1.0 / sqrt(ccf_sum)


def FWHM(rv, ccf, eccf=None, **kwargs):
    """
    Calculate the full width at half maximum (FWHM) of the CCF.
    
    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.
    kwargs : dict
        Keyword arguments passed directly to gaussfit
    """
    _, _, sig, _ = gaussfit(rv, ccf, yerr=eccf, **kwargs)
    return sig2fwhm(sig)


def FWHMerror(rv, ccf, eccf):
    """
    Calculate the uncertainty on the FWHM, following the same steps as the
    ESPRESSO DRS pipeline.

    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.
    eccf : array
        The uncertainties on each value of the CCF profile.
    """
    return 2.0 * RVerror(rv, ccf, eccf)


def contrast(rv, ccf, eccf=None, error=False, **kwargs):
    """
    Calculate the contrast (depth, measured in percentage) of the CCF.
    
    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.
    eccf : array
        The uncertainties on each value of the CCF profile.
    """
    A, _, _, continuum = gaussfit(rv, ccf, yerr=eccf, **kwargs)
    if not error:
        return abs(100 * A / continuum)
    else:
        ## error propagation
        # (A, _, _, continuum), (sA, _, _, sc) = gaussfit(rv, ccf, return_errors=True)
        # return abs(100 * (A / continuum) * np.hypot(sA / A, sc / continuum))
        ## as in the ESPRESSO pipeline
        fwhm = FWHM(rv, ccf, eccf, **kwargs)
        snr = RVerror(rv, ccf, eccf)  # 1.0 / sqrt(ccf_sum)
        return abs(snr * 2 / fwhm * A)


def mexican_hat(x, p):
    """ A Mexican Hat function with parameters p = [A, x0, σ, offset]. """
    _x = (x - p[1]) / p[2]
    return p[0] * (1 - _x**2) * np.exp(-_x**2 / 2) + p[3]

def _mexican_hat_partial_deriv(x, p):
    A, x0, sig, _ = p
    mh = mexican_hat(x, [A, x0, sig, 0.0])
    dA = mexican_hat(x, [1.0, x0, sig, 0.0])
    f = ((x - x0) / sig**2) * (3*sig**2 - (x - x0)**2) / (sig**2 - (x - x0)**2)
    dx0 = mh * f
    dsig = mh * (x - x0) * f / sig
    doffset = np.ones_like(x)
    return np.c_[dA, dx0, dsig, doffset]


def mexican_hat_fit(x: np.ndarray,
                    y: np.ndarray,
                    p0: Optional[List] = None,
                    yerr: Optional[np.ndarray] = None,
                    return_errors: bool = False,
                    use_deriv: bool = True,
                    guess_rv: Optional[float] = None,
                    **kwargs) -> List:
    """
    Fit a Mexican Hat function to `x`,`y` (and, if provided, `yerr`) using
    least-squares, with initial guess `p0` = [A, x0, σ, offset]. If p0 is not
    provided, the function tries an educated guess, which might lead to bad
    results.

    Args:
        x (array):
            The independent variable where the data is measured
        y (array):
            The dependent data.
        p0 (list or array):
            Initial guess for the parameters. If None, try to guess them from x,y.
        return_errors (bool):
            Whether to return estimated errors on the parameters.
        use_deriv (bool):
            Whether to use partial derivatives of the Mexican Hat (wrt the parameters)
            as Jacobian in the fit. If False, the Jacobian will be estimated.
        guess_rv (float):
            Initial guess for the RV (x0)
        **kwargs:
            Keyword arguments passed to `scipy.optimize.curve_fit`

    Returns:
        p (array):
            Best-fit values of the four parameters [A, x0, σ, offset]
        err (array):
            Estimated uncertainties on the four parameters
            (only if `return_errors=True`)
    """
    if (y == 0).all():
        return np.full(4, np.nan)

    x = x.astype(np.float64)
    y = y.astype(np.float64)

    if p0 is None:
        p0 = _gauss_initial_guess(x, y)
        p0[1] += 1e-5
        p0[2] *= 2.0

    if guess_rv is not None:
        p0[1] = guess_rv

    f = lambda x, *p: mexican_hat(x, p)
    if use_deriv:
        df = lambda x, *p: _mexican_hat_partial_deriv(x, p)
    else:
        df = None

    pfit, pcov, *_ = optimize.curve_fit(f, x, y, p0=p0, sigma=yerr, jac=df,
                                        xtol=1e-7, check_finite=True, **kwargs)

    if 'full_output' in kwargs:
            infodict = _

    if return_errors:
        errors = np.sqrt(np.diag(pcov))
        if 'full_output' in kwargs:
            return pfit, errors, infodict
        return pfit, errors

    if 'full_output' in kwargs:
        return pfit, infodict

    return pfit


def rv_shift(rv, ccf, radial_velocity):
    """ Shift a CCF profile by `radial_velocity`. """
    return CubicSpline(rv, ccf)(rv - radial_velocity)