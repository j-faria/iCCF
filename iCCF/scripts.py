""" Command-line scripts for iCCF """

import numpy as np
import matplotlib.pyplot as plt

import sys
import argparse

from . import iCCF

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
    parser.add_argument('--hdu', nargs=1, type=int,
                        help='HDU number')
    parser.add_argument('--sort', action='store_true', default=True,
                        help='sort the output by the MJD-OBS keyword')
    # parser.add_argument('--code', nargs=1, type=str,
    #                     help='code to generate "theoretical" samples '\
    #                          'to compare to the prior. \n'\
    #                          'Assign samples to an iterable called `samples`. '\
    #                          'Use numpy and scipy.stats as `np` and `st`, respectively. '\
    #                          'Number of prior samples in sample.txt is in variable `nsamples`. '\
    #                          'For example: samples=np.random.uniform(0,1,nsamples)')

    args = parser.parse_args()
    return args


def fits_to_rdb():
    args = _parse_args_fits_to_rdb()
    hdu_number = args.hdu[0]
    files = [line.strip() for line in sys.stdin]
    iCCF.indicators_from_files(files, hdu_number=hdu_number, sort_bjd=args.sort)