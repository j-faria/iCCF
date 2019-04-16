""" Helper functions to write to files and other outputs """

import numpy as np
import os

# standard_names = {
#     'vrad': ('RV', 'vrad'),
#     'fwhm': ('fwhm',),
#     'contrast': ('contrast',),
#     'bis': ('bis', 'bis_span', 'bisector'),
#     'wspan': ('wspan', 'w_span'),
#     'vspan': ('vspan', 'v_span'),
# }


def to_dict(I):
    """ Write the activity indicators inside `I` to a dictionary """
    d = {
        'vrad': I.RV,
        'svrad': I.RVerror,
        'fwhm': I.FWHM,
        'contrast': I.contrast,
        'bis_span': I.BIS,
        'wspan': I.Wspan,
        'vspan': I.Vspan,
    }
    try:
        d['DATE-OBS'] = I.HDU[0].header['DATE-OBS']
    except AttributeError:
        pass
    try:
        d['MJD-OBS'] = I.HDU[0].header['MJD-OBS']
    except AttributeError:
        pass

    return d


def to_rdb(I, filename='stdout', clobber=False):
    """ 
    Write the activity indicators in `I` to a .rdb file or stdout. If `clobber`
    is True, overwrite filename if it exists. 
    """
    cols = [
        'jdb',
        'vrad',
        'svrad',
        'fwhm',
        'contrast',
        'bis_span',
        'wspan',
        'vspan',
    ]

    header = '\t'.join(cols)
    header += '\n'
    header += '\t'.join('-' * len(c) for c in cols)

    vals = (
        I.HDU[0].header['MJD-OBS'],
        I.RV,
        I.RVerror,
        I.FWHM,
        I.contrast,
        I.BIS,
        I.Wspan,
        I.Vspan,
    )

    # output format
    fmt = ['%.6f']  # bjd is special
    fmt += (len(vals) - 1) * ['%%.%df' % I._nEPS]

    if filename == 'stdout':
        fmt = [f.replace('%', '') for f in fmt]
        print(header)
        print('\t'.join(('{0:%s}' % f).format(v) for v, f in zip(vals, fmt)))
    else:
        if os.path.exists(filename):
            if clobber:
                pass
            else:
                print('File "%s" exists, not overwriting without clobber=True.'
                      % filename)
                return

        np.savetxt(filename, np.c_[vals], header=header, delimiter='\t',
                   comments='', fmt=fmt)
