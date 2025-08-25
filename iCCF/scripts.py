""" Command-line scripts for iCCF """

import numpy as np
import matplotlib.pyplot as plt

import sys
from glob import glob
import argparse
import textwrap

from . import iCCF
from . import meta
from .utils import get_ncores
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
        iCCF.indicators_from_files(files, hdu_number=hdu_number,
                                   sort_bjd=args.sort, BIS_HARPS=bisHARPS)


desc_iccf_make_ccf = \
"""This script takes a list of S2D fits files and calculates the CCF for a
given RV array and a given mask. If these are not provided, it uses the same as
specified in the S2D file."""

def _parse_args_make_CCF():
    parser = argparse.ArgumentParser(
        prog='iccf-make-ccf',
        description='\n'.join(textwrap.wrap(desc_iccf_make_ccf)),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('files', nargs='+', type=str, help='S2D files')

    parser.add_argument('-o', '--output', type=str, help='Output file name')

    help_mask = 'Mask (G2, G9, K6, M2, ...). '\
                'A file called `INST_[mask].fits` should exist.'
    parser.add_argument('-m', '--mask', type=str, help=help_mask)

    parser.add_argument('-rv', '--rv', type=str,
                        help='RV array, in the form start:end:step [km/s]')
    parser.add_argument('--rv-range', type=float,
                        help='Full RV range where to calculate CCF [km/s]')

    parser.add_argument('--keep-prefix', action='store_true',
                        help='Keep any prefix of the S2D file on the output file')


    default_ncores = get_ncores()
    help_ncores = 'Number of cores to distribute calculation; '\
                  f'default is all available ({default_ncores})'
    parser.add_argument('--ncores', type=int, help=help_ncores)

    # help_ssh = 'An SSH user and host with which the script will try to find' \
    #            'required calibration files. It uses the `locate` and `scp`' \
    #            'commands to find and copy the file from the host'
    # parser.add_argument('--ssh', type=str, metavar='user@host', help=help_ssh)

    args = parser.parse_args()
    return args, parser


def make_CCF():
    args, _ = _parse_args_make_CCF()

    if sys.stdin.isatty():
        files = args.files
        # print('pipe something (a list of S2D fits files) into this script')
        # print('example: ls *S2D* | iccf-make-ccf [options]')
        # sys.exit(1)
    else:
        files = [line.strip() for line in sys.stdin]

    if len(files) == 1 and '*' in files[0]:
        files = glob(files[0])

    for file in files:
        print('Calculating CCF for', file)
        header = fits.open(file)[0].header

        if args.rv is None:
            try:
                try:
                    OBJ_RV = header['HIERARCH ESO OCS OBJ RV']
                except KeyError:
                    OBJ_RV = header['HIERARCH ESO TEL TARG RADVEL']
                start = header['HIERARCH ESO RV START']
                step = header['HIERARCH ESO RV STEP']

                if args.rv_range is not None:
                    start = OBJ_RV - float(args.rv_range)
                    end = OBJ_RV + float(args.rv_range)
                else:
                    end = OBJ_RV + (OBJ_RV - start)

                print('Using RV array from S2D file:',
                      f'{start} : {end} : {step} km/s')
                rvarray = np.arange(start, end + step, step)
            except KeyError:
                print('Could not find RV start and step in S2D file. Please use the -rv argument.')
                sys.exit(1)
        else:
            args.rv = args.rv.replace('"', '').replace("'", '')
            start, end, step = map(float, args.rv.split(':'))
            rvarray = np.arange(start, end + step, step)
            print(f'Using RV array: {start} : {end} : {step} km/s')

        inst = header['INSTRUME']

        mask = args.mask
        if mask is None:
            try:
                mask = header['HIERARCH ESO QC CCF MASK']
            except KeyError:
                try:
                    mask = header['HIERARCH ESO PRO REC1 CAL25 NAME']
                    if inst + '_' in mask:
                        mask = mask[9:11]
                except KeyError:
                    print('Could not find CCF mask in S2D file. Please use the -m argument.')
                    sys.exit(1)
            print('Using mask from S2D file:', mask)

            # if not os.path.exists(f'{inst}_{mask}.fits'):
            #     print(f'File "{inst}_{mask}.fits" not found.')
            #     sys.exit(1)

        meta.calculate_ccf(file, mask=mask, rvarray=rvarray,
                           ncores=args.ncores, output=args.output,
                           keep_prefix=args.keep_prefix,
                           # ssh=args.ssh
                           )


def _parse_args_check_CCF():
    parser = argparse.ArgumentParser(
        prog='iccf-check-ccf',
    )

    parser.add_argument('file1', type=str, help='original CCF file')
    parser.add_argument('file2', type=str, help='new CCF file')
    parser.add_argument('-c', '--compact', action='store_true',
                        help='only show difference in RV')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='plot each CCF and their difference')

    args = parser.parse_args()
    return args, parser

def print_difference_caret(v1, v2, space='     '):
    a, b = str(v1), str(v2)
    if a != b:
        try:
            size = [i for i,(_a, _b) in enumerate(zip(a, b)) if _a != _b][0]
            print(space + (' ' * (size+1)) + '^')
        except IndexError:
            pass

def check_CCF(file1=None, file2=None, compact=False, plot=False):
    from .utils import float_exponent

    if file1 is None and file2 is None:
        args, _ = _parse_args_check_CCF()
    else:
        args = argparse.Namespace(file1=file1, file2=file2,
                                  compact=compact, plot=plot)

    if not args.compact:
        print('Comparing CCFs from')
        print('  ', args.file1)
        print('  ', args.file2)
        print()

    i1 = iCCF.Indicators.from_file(args.file1)
    i2 = iCCF.Indicators.from_file(args.file2)

    if args.compact:
        print(' km  m cm')
        print(f'{i1.RV:<15.10f} - {args.file1}')
        print(f'{i2.RV:<15.10f} - {args.file2}')
        print_difference_caret(i1.RV, i2.RV, space='')
        return

    try:
        print('Absolute differences:')
        a = i1._SCIDATA - i2._SCIDATA
        print(f'CCF1 - CCF2       : max={np.nanmax(a):.2e}, mean={np.nanmean(a):.2e}')
        print('Relative differences:')
        with np.errstate(invalid='ignore', divide='ignore'):
            r = a / i1._SCIDATA
        print(f'(CCF1 - CCF2)/CCF1: max={np.nanmax(r):.2e}, mean={np.nanmean(r):.2e}')
        print()
    except ValueError:
        print('Could not compare CCFs')

    print()
    print('  RV:', i1.RV, '(pipe RV=', i1.pipeline_RV, ')')
    print('    :', i2.RV)
    print_difference_caret(i1.RV, i2.RV)

    print()
    print('FWHM:', i1.FWHM, '(pipe FWHM=', i1.pipeline_FWHM, ')')
    print('    :', i2.FWHM)
    a, b = str(i1.FWHM), str(i2.FWHM)
    if a != b:
        try:
            size = [i for i,(_a, _b) in enumerate(zip(a, b)) if _a != _b][0]
            print('     ' + (' ' * (size+1)) + '^')
        except IndexError:
            pass

    if args.plot:
        if i1.rv.size != i2.rv.size:
            print('Warning: RV arrays are not the same size')
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            i1.plot(ax)
            i2.plot(ax)
        else:
            # fig, axs = plt.subplots(2, 1, constrained_layout=True, height_ratios=(3, 1))
            fig, axs = plt.subplot_mosaic('ac\nbd', constrained_layout=True, height_ratios=(3, 1))
            i1.plot(axs['a'])
            i2.plot(axs['a'])
            axs['a'].legend().remove()
            axs['b'].plot(i1.rv, i1.ccf - i2.ccf, 'k.')
            axs['b'].set(xlabel='RV [km/s]', ylabel='CCF difference')

            i1.plot_individual_CCFs(ax=axs['c'], color='C0', alpha=0.6)
            i2.plot_individual_CCFs(ax=axs['c'], color='C1', alpha=0.6)
            diff = i1._SCIDATA - i2._SCIDATA
            x = i1._get_x_for_plot_individual_CCFs()
            for _x, _diff in zip(x, diff):
                axs['d'].plot(_x, _diff, 'k', alpha=0.6)
            axs['d'].set(xlabel='spectral order', ylabel='CCF difference')

            axs['b'].sharex(axs['a'])
            axs['d'].sharex(axs['c'])

        plt.show()
        return fig

def _parse_args_recreate_CCF():
    parser = argparse.ArgumentParser(prog='iccf-recreate-ccf')
    parser.add_argument('file', type=str, help='CCF file')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite output file if it already exists')
    args = parser.parse_args()
    return args, parser

def recreate_CCF():
    args, _ = _parse_args_recreate_CCF()
    from iCCF.iCCF import recreate_ccf_file
    recreate_ccf_file(args.file, overwrite=args.overwrite)