import numpy as np 
from numpy import exp, log, sqrt, inf
from scipy import optimize

from .utils import numerical_gradient

def gauss(x, p):
    """ A Gaussian function with parameters p = [A, x0, σ, offset]. """
    return p[0] * exp(-(x - p[1])**2 / (2 * p[2]**2)) + p[3]


def _gauss_partial_deriv(x, p):
    """ Partial derivatives of a Gaussian with respect to each parameter. """
    A, x0, sig, _ = p
    g = gauss(x, [A, x0, sig, 0.0])
    dgdA = gauss(x, [1.0, x0, sig, 0.0])
    dgdx0 = g * ((x-x0)/sig**2)
    dgdsig = g * ((x-x0)**2/sig**3)
    dgdoffset = np.ones_like(x)
    return np.c_[dgdA, dgdx0, dgdsig, dgdoffset]


def _gauss_initial_guess(x, y):
    """ Educated guess (from the data) for Gaussian parameters. """
    p0 = []
    p0.append(y.mean() - y.max())  # guess the amplitude
    # guess the center, but maybe the CCF is upside down?
    if y[x.size // 2] > y[0]:  # seems like it
        p0.append(x[y.argmax()])
    else:
        p0.append(x[y.argmin()])
    p0.append(1)  # guess the sigma
    p0.append(y.mean())  # guess the offset 
    return p0

def gaussfit(x, y, p0=None, return_errors=False, use_deriv=True):
    """ 
    Fit a Gaussian function to `x`,`y` using least-squares, with initial guess
    `p0` = [A, x0, σ, offset]. If p0 is not provided, the function tries an
    educated guess, which might lead to bad results.

    Parameters
    ----------
    x : array
        The independent variable where the data is measured
    y : array
        The dependent data.
    p0 : list or array
        Initial guess for the parameters. If None, try to guess them from x,y.
    return_errors : bool
        Whether to return estimated errors on the parameters.
    use_deriv : bool
        Whether to use partial derivatives of the Gaussian (wrt the parameters)
        as Jacobian in the fit. If False, the Jacobian will be estimated.
    """
    if (y == 0).all():
        return np.nan * np.ones(4)

    f = lambda p, x, y: gauss(x, p) - y
    if use_deriv:
        df = lambda p, x, y: _gauss_partial_deriv(x, p)
    else:
        df = None
    
    if p0 is None:
        p0 = _gauss_initial_guess(x, y)
    
    result = optimize.leastsq(f, p0, args=(x, y), Dfun=df, full_output=True)
    pfit, pcov, infodict, errmsg, success = result

    pfit[2] = abs(pfit[2]) # positive sigma

    if return_errors:
        s_sq = (f(pfit, x, y)**2).sum() / (y.size - pfit.size)
        pcov = pcov * s_sq
        errors = np.sqrt(np.diag(pcov))
        return pfit, errors
    else:
        return pfit
    # bounds = ([-inf, -inf, 0, -inf], [inf, inf, inf, inf])
    # return optimize.curve_fit(f, x, y, p0=p0, bounds=bounds)[0]


def sig2fwhm(sig):
    """ Convert standard deviation to full width at half maximum. """
    return 2 * sqrt(2 * log(2)) * sig


def fwhm2sig(fwhm):
    """ Convert full width at half maximum to standard deviation. """
    return fwhm / (2 * sqrt(2 * log(2)))


def RV(rv, ccf, **kwargs):
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
    _, rv, _, _ = gaussfit(rv, ccf, **kwargs)
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


def FWHM(rv, ccf):
    """
    Calculate the full width at half maximum (FWHM) of the CCF.
    
    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.
    """
    _, _, sig, _ = gaussfit(rv, ccf)
    return sig2fwhm(sig)


def contrast(rv, ccf):
    """
    Calculate the contrast (depth, measured in percentage) of the CCF.
    
    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.
    """
    A, _, _, continuum = gaussfit(rv, ccf)
    return abs(100 * A / continuum)
