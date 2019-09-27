""" Command-line scripts for iCCF """

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import argparse

from . import iCCF
from . import meta_ESPRESSO
# from .meta_ESPRESSO import calculate_ccf as calculate_ccf_ESPRESSO
from astropy.io import fits


def _parse_args_fits_to_rdb():
    desc = """
    This script takes a list of CCF fits files and outputs CCF activity 
    indicators to stdout, in rdb format.
    """
    parser = argparse.ArgumentParser(
        description=desc,
        prog='iccf-fits-to-rdb',
    )
    # parser.add_argument('column', nargs=1, type=int,
    #                     help='which column to use for histogram')
    parser.add_argument('--hdu', type=int, help='HDU number')
    parser.add_argument('--sort', action='store_true', default=True,
                        help='sort the output by the MJD-OBS keyword')
    parser.add_argument('--bis-harps', action='store_true', default=True,
                        help='do the bisector calculation as in HARPS')
    # parser.add_argument('--code', nargs=1, type=str,
    #                     help='code to generate "theoretical" samples '\
    #                          'to compare to the prior. \n'\
    #                          'Assign samples to an iterable called `samples`. '\
    #                          'Use numpy and scipy.stats as `np` and `st`, respectively. '\
    #                          'Number of prior samples in sample.txt is in variable `nsamples`. '\
    #                          'For example: samples=np.random.uniform(0,1,nsamples)')

    args = parser.parse_args()
    return args, parser


def fits_to_rdb():
    args, _ = _parse_args_fits_to_rdb()
    # print(args)

    bisHARPS = args.bis_harps
    hdu_number = args.hdu

    if sys.stdin.isatty():
        print('pipe something (a list of CCF fits files) into this script')
        sys.exit(1)
    else:
        files = [line.strip() for line in sys.stdin]
        # print(files)
        iCCF.indicators_from_files(
            files,
            hdu_number=hdu_number,
            sort_bjd=args.sort,
            BIS_HARPS=bisHARPS,
        )


default_ncores = len(os.sched_getaffinity(0))


def _parse_args_make_CCF():
    desc = """
    This script takes a list of S2D fits files and calculates the CCF for a 
    given RV array and a given mask. If no mask is provided, it uses the same as
    specified in the S2D file.
    """
    parser = argparse.ArgumentParser(
        description=desc,
        prog='iccf-make-ccf',
    )
    parser.add_argument('-m', '--mask', type=str, help='Mask (G2, G9, ...)')
    parser.add_argument('-rv', required=True, type=str,
                        help='RV array, in the form start:end:step')
    help_ncores = 'Number of cores to distribute calculation; '\
                  f'default is all available ({default_ncores})'
    parser.add_argument('--ncores', type=int, help=help_ncores)

    args = parser.parse_args()
    return args, parser


def make_CCF():
    args, _ = _parse_args_make_CCF()
    # print(args)

    start, end, step = map(float, args.rv.split(':'))
    rvarray = np.arange(start, end + step, step)

    if sys.stdin.isatty():
        print('pipe something (a list of S2D fits files) into this script')
        sys.exit(1)
    else:
        files = [line.strip() for line in sys.stdin]

        for file in files:
            print('Calculating CCF for', file)
            header = fits.open(file)[0].header

            mask = args.mask
            if mask is None:
                mask = header['HIERARCH ESO QC CCF MASK']

            inst = header['INSTRUME']

            if inst == 'ESPRESSO':
                meta_ESPRESSO.calculate_ccf(file, mask=mask, rvarray=rvarray,
                                            ncores=args.ncores)

            elif inst == 'HARPS':
                print('dont know what to do with HARPS! sorry')
