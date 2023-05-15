import warnings
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
    dgdx0 = g * ((x - x0) / sig**2)
    dgdsig = g * ((x - x0)**2 / sig**3)
    dgdoffset = np.ones_like(x)
    return np.c_[dgdA, dgdx0, dgdsig, dgdoffset]


def _gauss_initial_guess(x, y):
    """ Educated guess (from the data) for Gaussian parameters. """
    # these guesses tend to work better for narrow-ish gaussians
    p0 = []

    # guess the amplitude (force it to be negative)
    p0.append(-abs(y.ptp()))

    # guess the center, but maybe the CCF is upside down?
    m = y.mean()
    ups_down = np.sign(np.percentile(y, 50) - m) != np.sign(y.max() - m)
    if ups_down:  # seems like it
        # warnings.warn('It seems the CCF might be upside-down?')
        p0.append(x[y.argmax()])
    else:
        p0[0] *= -1
        p0.append(x[y.argmin()])
    # guess the width
    p0.append(1)
    # guess the offset
    p0.append(0.5 * (y[0] + y[-1]))

    return p0


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

    Returns:
        p (array):
            Best-fit values of the four parameters [A, x0, σ, offset]
        err (array):
            Estimated uncertainties on the four parameters
            (only if `return_errors=True`)
    """
    if (y == 0).all():
        return np.nan * np.ones(4)

    if yerr is None:
        f = lambda p, x, y: gauss(x, p) - y
    else:
        f = lambda p, x, y, yerr: ((gauss(x, p) - y) / yerr)

    if use_deriv:
        df = lambda p, x, *_: _gauss_partial_deriv(x, p)
    else:
        df = None

    if p0 is None:
        p0 = _gauss_initial_guess(x, y)
    if guess_rv is not None:
        p0[1] = guess_rv

    # if yerr is None:
    #     args = (x, y)
    # else:
    #     args = (x, y, yerr)

    # result = optimize.leastsq(f, p0, args=args, Dfun=df, full_output=True)
    # pfit, pcov, infodict, errmsg, success = result
    # # return result
    # pfit[2] = abs(pfit[2])  # positive sigma

    # if return_errors:
    #     if yerr is None:
    #         s_sq = (f(pfit, x, y)**2).sum() / (y.size - pfit.size)
    #     else:
    #         s_sq = (f(pfit, x, y, yerr)**2).sum() / (y.size - pfit.size)
    #     print(s_sq)
    #     pcov = pcov * s_sq
    #     errors = np.sqrt(np.diag(pcov))
    #     return pfit, errors
    # else:
    #     return pfit

    f = lambda x, *p: gauss(x, p)
    if use_deriv:
        df = lambda x, *p: _gauss_partial_deriv(x, p)
    else:
        df = None

    pfit, pcov = optimize.curve_fit(f, x, y, p0=p0, sigma=yerr, jac=df)

    if return_errors:
        errors = np.sqrt(np.diag(pcov))
        return pfit, errors
    return pfit


def sig2fwhm(sig):
    """ Convert standard deviation to full width at half maximum. """
    return 2 * sqrt(2 * log(2)) * sig


def fwhm2sig(fwhm):
    """ Convert full width at half maximum to standard deviation. """
    return fwhm / (2 * sqrt(2 * log(2)))


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
        The errors on each value of the CCF profile.
    """
    return 2.0 * RVerror(rv, ccf, eccf)


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
