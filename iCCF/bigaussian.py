from typing import List, Optional
import numpy as np
from scipy import optimize
from .gaussian import gauss, gaussfit, _gauss_initial_guess


def bigauss(x, p):
    """
    A bi-Gaussian function with parameters p = [A, x0, σ, asy, offset]. The
    `asy` parameter controls the asymmetry of the bi-Gaussian, given as a
    fraction of σ. A negative value for `asy` results in a left-skewed
    bi-Gaussian. For `asy` = 0, the bi-Gaussian is the same as a Gaussian.
    """
    σ1 = p[2] * (1 - p[3])
    σ2 = p[2] * (1 + p[3])
    return p[4] + np.piecewise(x, [x < p[1], x >= p[1]],
        [
            lambda x: gauss(x, [p[0], p[1], σ1, 0.0]), 
            lambda x: gauss(x, [p[0], p[1], σ2, 0.0])
        ]
    )

def _bigauss_partial_deriv(x, p):
    """ Partial derivatives of a bi-Gaussian with respect to each parameter. """
    A, x0, σ, asy, offset = p
    σ1 = σ * (1 - asy)
    σ2 = σ * (1 + asy)

    conds = [x < x0, x >= x0]
    g1 = lambda x: gauss(x, [A, x0, σ1, 0.0])
    g2 = lambda x: gauss(x, [A, x0, σ2, 0.0])

    ddA = np.piecewise(x, conds,
        [
            lambda x: gauss(x, [1.0, x0, σ1, 0.0]), 
            lambda x: gauss(x, [1.0, x0, σ2, 0.0])
        ]
    )
    ddx0 = np.piecewise(x, conds,
        [
            lambda x: ((x - x0) / σ1**2) * g1(x), 
            lambda x: ((x - x0) / σ2**2) * g2(x)
        ],
    )
    ddσ = np.piecewise(x, conds,
        [
            lambda x: ((x - x0)**2 / (σ * (σ1**2))) * g1(x), 
            lambda x: ((x - x0)**2 / (σ * (σ2**2))) * g2(x)
        ]
    )
    ddasy = np.piecewise(x, conds,
        [
            lambda x: ((x - x0)**2 / (σ1**2 * (asy - 1))) * g1(x), 
            lambda x: ((x - x0)**2 / (σ2**2 * (asy + 1))) * g2(x)
        ]
    )
    ddoffset = np.ones_like(x)

    return np.c_[ddA, ddx0, ddσ, ddasy, ddoffset]


def bigaussfit(x: np.ndarray,
               y: np.ndarray,
               p0: Optional[List] = None,
               yerr: Optional[np.ndarray] = None,
               return_errors: bool = False,
               use_deriv: bool = True,
               guess_rv: Optional[float] = None,
               **kwargs) -> List:
    """
    Fit a bi-Gaussian function to `x`,`y` (and, if provided, `yerr`) using
    least-squares, with initial guess `p0` = [A, x0, σ, asy, offset]. If p0 is
    not provided, the function tries an educated guess, which might lead to bad
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
            Whether to use partial derivatives of the bi-Gaussian (wrt the
            parameters) as Jacobian in the fit. If False, the Jacobian will be
            estimated.
        guess_rv (float):
            Initial guess for the RV (x0)
        **kwargs:
            Keyword arguments passed to `scipy.optimize.curve_fit`

    Returns:
        p (array):
            Best-fit values of the four parameters [A, x0, σ, asy, offset]
        err (array):
            Estimated uncertainties on the five parameters (only if
            `return_errors=True`)
    """
    if (y == 0).all():
        return 5 * [np.nan]

    x = x.astype(y.dtype.type)
    y = y.astype(y.dtype.type)

    if p0 is None:
        p0 = _gauss_initial_guess(x, y)
        p0.insert(3, 0.0)
    if guess_rv is not None:
        p0[1] = guess_rv

    p0 = np.array(p0).astype(y.dtype.type)

    f = lambda x, *p: bigauss(x, p)
    if use_deriv:
        df = lambda x, *p: _bigauss_partial_deriv(x, p)
    else:
        df = None

    # bounds = ([-np.inf]*4, [np.inf]*4)
    # bounds[0][1] = x.min()
    # bounds[1][1] = x.max()
    # print(p0)
    # print(bounds)

    pfit, pcov, *_ = optimize.curve_fit(f, x, y, p0=p0, sigma=yerr, jac=df,
                                        xtol=1e-12, ftol=1e-14, 
                                        check_finite=True, **kwargs)

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


def DeltaV(rv, ccf, eccf=None, **kwargs):
    """
    Calculate the ΔV indicator, defined in Figueira et al. (2013, A&A, 557,
    A93). This represents the RV shift that can be explained by the line profile
    asymmetry alone.

    Args:
        rv (ndarray):
            The velocity values where the CCF is defined.
        ccf (ndarray):
            The values of the CCF profile.
        eccf (ndarray):
            The errors on each value of the CCF profile.
    """
    # fit a Gaussian function to the CCF and extract the RV
    p_G = gaussfit(rv, ccf, yerr=eccf, **kwargs)
    RV_G = p_G[1]
    # guess for the bi-Gaussian asymmetry is 0
    p_G = list(p_G)
    p_G.insert(3, 0.0)
    # fit a bi-Gaussian function to the CCF and extract the corresponding RV
    p_biG = bigaussfit(rv, ccf, yerr=eccf, p0=p_G, **kwargs)
    RV_biG = p_biG[1]
    # DeltaV is the difference between the two RV values
    return RV_biG - RV_G
