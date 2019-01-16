import os
import re
import numpy as np

# from https://stackoverflow.com/a/30141358/1352183
def running_mean(x, N=2):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


# from https://stackoverflow.com/a/16090640
def natsort(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def numerical_gradient(rv, ccf):
    """
    Return the gradient of the CCF.

    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.

    Notes
    -----
    The gradient is computed using the np.gradient routine, which uses second
    order accurate central differences in the interior points and either first
    or second order accurate one-sides (forward or backwards) differences at
    the boundaries. The gradient has the same shape as the input array.
    """
    return np.gradient(ccf, rv)


def find_myself():
    """ Return the path to iCCF's source. """
    thisdir = os.path.dirname(os.path.realpath(__file__))
    topdir = os.path.dirname(thisdir)
    return topdir


def load_example_data():
    """ Load the example CCF stored in iCCF/example_data """
    topdir = find_myself()
    return np.load(os.path.join(topdir, 'example_data', 'CCF1.npy'))
