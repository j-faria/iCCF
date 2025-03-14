""" Calculate CCFs themselves. So meta! """

import multiprocessing
import os
import subprocess
import sys
import time as pytime
import warnings
from bisect import bisect_left
from glob import glob
from itertools import product

import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from tqdm import tqdm

from iCCF.keywords import getINSTRUMENT

from .iCCF import Indicators
from .masks import Mask
from .utils import get_ncores, find_data_file, disable_exception_traceback


def espdr_compute_CCF_fast(ll, dll, flux, error, blaze, quality, RV_table,
                           mask, berv, bervmax, mask_width=0.5):
    c = 299792.458

    nx_s2d = flux.size
    # ny_s2d = 1  #! since this function computes only one order
    n_mask = mask.size
    nx_ccf = len(RV_table)

    ccf_flux = np.zeros_like(RV_table)
    ccf_error = np.zeros_like(RV_table)
    ccf_quality = np.zeros_like(RV_table)

    dll2 = dll / 2.0  # cpl_image_divide_scalar_create(dll,2.);
    ll2 = ll - dll2  # cpl_image_subtract_create(ll,dll2);

    #? this mimics the pipeline (note that cpl_image_get indices start at 1)
    imin, imax = 1, nx_s2d
    while(imin < nx_s2d and quality[imin-1] != 0):
        imin += 1
    while(imax > 1 and quality[imax-1] != 0):
        imax -= 1

    if imin >= imax:
        return
    #? note that cpl_image_get indices start at 1, hence the "-1"s
    llmin = ll[imin + 1 - 1] / (1. + berv / c) * (1. + bervmax / c) / (1. + RV_table[0] / c)
    llmax = ll[imax - 1 - 1] / (1. + berv / c) * (1. - bervmax / c) / (1. + RV_table[nx_ccf - 1] / c)

    imin, imax = 0, n_mask - 1

    #? turns out cpl_table_get indices start at 0...
    while (imin < n_mask and mask['lambda'][imin] < (llmin + 0.5 * mask_width / c * llmin)):
        imin += 1
    while (imax >= 0     and mask['lambda'][imax] > (llmax - 0.5 * mask_width / c * llmax)):
        imax -= 1

    for i in range(imin, imax + 1):
        #? cpl_array_get indices also start at 0
        llcenter = mask['lambda'][i] * (1. + RV_table[nx_ccf // 2] / c)

        # index_center = 1
        # while(ll[index_center-1] < llcenter): index_center += 1
        # my attempt to speed it up
        # index_center = np.where(ll < llcenter)[0][-1] + 1
        index_center = bisect_left(ll, llcenter) + 1

        contrast = mask['contrast'][i]
        w = contrast * contrast

        for j in range(0, nx_ccf):
            llcenter = mask['lambda'][i] * (1. + RV_table[j] / c)
            llstart = llcenter - 0.5 * mask_width / c * llcenter
            llstop = llcenter + 0.5 * mask_width / c * llcenter

            # index1 = 1
            # while(ll2[index1-1] < llstart): index1 += 1
            index1 = bisect_left(ll2, llstart) + 1

            # index2 = index1
            # while (ll2[index2-1] < llcenter): index2 += 1
            index2 = bisect_left(ll2, llcenter) + 1

            # index3 = index2
            # while (ll2[index3-1] < llstop): index3 += 1;
            index3 = bisect_left(ll2, llstop) + 1

            k = j

            for index in range(index1, index3):
                ccf_flux[k] += w * flux[index-1] / blaze[index-1] * blaze[index_center-1]  # noqa: E501

            ccf_flux[k] += w * flux[index1-1-1] * (ll2[index1-1]-llstart) / dll[index1-1-1] / blaze[index1-1-1] * blaze[index_center-1]
            ccf_flux[k] -= w * flux[index3-1-1] * (ll2[index3-1]-llstop) / dll[index3-1-1] / blaze[index3-1-1] * blaze[index_center-1]

            ccf_error[k] += w * w * error[index2 - 1 - 1] * error[index2 - 1 - 1]

            ccf_quality[k] += quality[index2 - 1 - 1]

    # my_error = cpl_image_power(*CCF_error_RE,0.5);
    ccf_error = np.sqrt(ccf_error)

    return ccf_flux, ccf_error, ccf_quality


def check_if_deblazed(s2dfile):
    if 'S2D_BLAZE' in s2dfile:
        return False
    if 'S2D_A.fits' in s2dfile:
        return True
    


def find_blaze(s2dfile, hdu=None):
    if hdu is None:
        header = fits.getheader(s2dfile)
    else:
        header = hdu[0].header

    for k, v in dict(header['*CAL* CATG']).items():
        if v == 'BLAZE_A':
            return k, header[k.replace('CATG', 'NAME')]


def calculate_s2d_ccf(s2dfile, rvarray, order='all',
                      mask_file='ESPRESSO_G2.fits', mask=None, mask_width=0.5,
                      debug=False):

    with fits.open(s2dfile) as hdu:

        if order == 'all':
            if debug:
                print('can only debug one order at a time...')
                return
            orders = range(hdu[1].data.shape[0])
            return_sum = True
        else:
            assert isinstance(order, int), 'order should be integer'
            orders = (order, )
            return_sum = False

        BERV = hdu[0].header['HIERARCH ESO QC BERV']
        BERVMAX = hdu[0].header['HIERARCH ESO QC BERVMAX']

        dllfile = hdu[0].header['HIERARCH ESO PRO REC1 CAL7 NAME']
        blazefile = hdu[0].header['HIERARCH ESO PRO REC1 CAL13 NAME']
        print('need', dllfile)
        print('need', blazefile)

        dllfile = glob(dllfile + '*')[0]

        # CCF mask
        if mask is None:
            with fits.open(mask_file) as mask_hdu:
                mask = mask_hdu[1].data
        else:
            assert 'lambda' in mask, 'mask must contain the "lambda" key'
            assert 'contrast' in mask, 'mask must contain the "contrast" key'

        # get the flux correction stored in the S2D file
        keyword = 'HIERARCH ESO QC ORDER%d FLUX CORR'
        flux_corr = [hdu[0].header[keyword % (o + 1)] for o in range(170)]

        ccfs, ccfes = [], []

        with fits.open(dllfile) as hdu_dll:
            dll_array = hdu_dll[1].data

        with fits.open(blazefile) as hdu_blaze:
            blaze_array = hdu_blaze[1].data

        for order in orders:
            # WAVEDATA_AIR_BARY
            ll = hdu[5].data[order, :]
            # mean w
            llc = np.mean(hdu[5].data, axis=1)

            dll = dll_array[order, :]
            # fit an 8th degree polynomial to the flux correction
            corr_model = np.polyval(np.polyfit(llc, flux_corr, 7), llc)

            flux = hdu[1].data[order, :]
            error = hdu[2].data[order, :]
            quality = hdu[3].data[order, :]

            blaze = blaze_array[order, :]

            y = flux * blaze / corr_model[order]
            # y = np.loadtxt('flux_in_pipeline_order0.txt')
            ye = error * blaze / corr_model[order]

            if debug:
                return ll, dll, y, ye, blaze, quality, rvarray, mask, BERV, BERVMAX

            print('calculating ccf (order %d)...' % order)
            ccf, ccfe, _ = espdr_compute_CCF_fast(ll, dll, y, ye, blaze, quality,
                                                  rvarray, mask, BERV, BERVMAX,
                                                  mask_width=mask_width)
            ccfs.append(ccf)
            ccfes.append(ccfe)

    if return_sum:
        ccf = np.concatenate([ccfs, np.array(ccfs).sum(axis=0, keepdims=True)])
        ccfe = np.concatenate([ccfes, np.zeros(len(rvarray)).reshape(1, -1)])
        # what to do with the errors?
        return ccf, ccfe
    else:
        return np.array(ccfs), np.array(ccfes)


def find_file(file, ssh=None, verbose=True):
    if verbose:
        print('Looking for file:', file)

    # first try here:
    if os.path.exists(file) or os.path.exists(file + '.fits'):
        if verbose:
            print('\tfound it in current directory')
        return glob(file + '*')[0]

    similar = glob(file + '*.fits')
    if len(similar) > 0:
        file = similar[0]
        if verbose:
            print(f'\tfound a similar file in current directory ({file})')
        return file

    # try in iCCF/data
    try:
        f = find_data_file(file)
        if verbose:
            print('\tfound file in iCCF/data')
        return f
    except FileNotFoundError:
        pass

    # try on the local machine
    try:
        found = subprocess.check_output(f'locate {file}'.split())
        found = found.decode().split()
        if len(found) == 0:
            raise FileNotFoundError(file)
        if verbose:
            print('\tfound file:', found[-1])
        return found[-1]
    except (subprocess.CalledProcessError, FileNotFoundError):
        if ssh is None:
            raise FileNotFoundError(file) from None

    # try on a server with SSH
    if ssh is not None:
        if '@' not in ssh:
            raise ValueError('ssh should be in the form "user@host"')
        # user, host = ssh.split('@')
        locate_cmd = f'ssh {ssh} locate {file}'
        try:
            found = subprocess.check_output(locate_cmd.split())
            found = found.decode().split()
            if verbose:
                print('\tfound file:', ssh + ':' + found[-1])
        except subprocess.CalledProcessError:
            raise FileNotFoundError(file) from None

        full_path = found[-1]
        scp_cmd = f'scp {ssh}:{full_path} .'
        try:
            subprocess.check_call(scp_cmd.split())
            return os.path.split(full_path)[-1]
        except subprocess.CalledProcessError:
            raise RuntimeError(f'Could not scp {file} from {ssh}') from None



def _dowork(args):
    order, kwargs = args
    data = kwargs['data']
    dll = kwargs['dll'][order]
    blaze = kwargs['blaze'][order]
    corr_model = kwargs['corr_model']
    rvarray = kwargs['rvarray']
    mask = kwargs['mask']
    BERV = kwargs['BERV']
    BERVMAX = kwargs['BERVMAX']
    mask_width = kwargs['mask_width']

    # WAVEDATA_AIR_BARY
    ll = data[5][order, :]

    flux = data[1][order, :]
    error = data[2][order, :]
    quality = data[3][order, :]

    y = flux * blaze / corr_model[order]
    ye = error * blaze #/ corr_model[order]

    ccf, ccfe, ccfq = espdr_compute_CCF_fast(ll, dll, y, ye, blaze, quality, rvarray, mask,
                                             BERV, BERVMAX, mask_width=mask_width)

    assert np.isfinite(ccf).all()
    assert np.isfinite(ccfe).all()

    return ccf, ccfe, ccfq


def calculate_s2d_ccf_parallel(s2dfile, rvarray, mask, mask_width=0.5, order='all',
                               ncores=None, verbose=True, full_output=False,
                               smart_blaze=True, ignore_blaze=False, do_flux_corr=False,
                               ssh=None):
    """
    Calculate the CCF between a 2D spectra and a mask. This function can lookup
    necessary files (locally or over SSH) and can perform the calculation in
    parallel, depending on the value of `ncores`

    Arguments
    ---------
    s2dfile : str
        The name of the S2D file
    rvarray : array
        RV array where to calculate the CCF
    order : str or int
        Either 'all' to calculate the CCF for all orders, or the order
    mask : str
        An object with 'lambda' and 'contrast' keys corresponding to the CCF
        mask lines.
    mask_width : float
        The width of the mask "lines" in km/s
    ncores : int
        Number of CPU cores to use for the calculation (default: all available)
    verbose : bool, default True
        Print messages and a progress bar during the calculation
    full_output : bool, default False
        Return all the quantities that went into the CCF calculation (some 
        extracted from the S2D file)
    smart_blaze : bool, default True
        If True, the function will try to reconstruct the blaze file by looking
        for the corresponding '*S2D_BLAZE*' file
    ignore_blaze : bool, default True
        If True, the function completely ignores any blaze correction and takes
        the flux values as is from the S2D file
    do_flux_corr : bool
        Whether to perform the flux correction step. By default, check the 
        'HIERARCH ESO QC SCIRED FLUX CORR CHECK' keyword in the S2D header
    ssh : str
        SSH information in the form "user@host" to look for required calibration
        files in a server. If the files are not found locally, the function
        tries the `locate` and `scp` commands to find and copy the file from the
        SSH host
    """
    hdu = fits.open(s2dfile)
    norders, order_len = hdu[1].data.shape

    if ncores is None:
        ncores = get_ncores() // 2

    if verbose:
        print(f'Using {ncores} CPU cores for the calculation')

    if order == 'all':
        orders = range(hdu[1].data.shape[0])
        return_sum = True
        if verbose:
            pass
    else:
        assert isinstance(order, int), 'order should be integer'
        orders = (order, )
        return_sum = False
        if verbose:
            pass

    BERV = hdu[0].header['HIERARCH ESO QC BERV']
    BERVMAX = hdu[0].header['HIERARCH ESO QC BERVMAX']

    if verbose:
        print(f'Mask width: {mask_width}')
        print(f'BERVMAX: {BERVMAX} km/s')
        print(f'BERV: {BERV} km/s')

    # find and read the blaze file
    if ignore_blaze:
        blaze = np.ones_like(hdu[1].data)
    else:
        _, blazefile = find_blaze(None, hdu=hdu)
        try:
            blazefile = find_file(blazefile, ssh, verbose)
        except FileNotFoundError:
            for replacement in [(':', '_'), (':', '%3A')]:
                try:
                    blazefile = find_file(blazefile.replace(*replacement), ssh, verbose)
                    break
                except FileNotFoundError:
                    pass
            else:
                # print(f'Could not find blaze file "{blazefile}"')
                # sys.exit(1)
                if smart_blaze:
                    if verbose:
                        print('Could not find blaze file. Trying smart blaze')
                else:
                    raise FileNotFoundError(f'Could not find BLAZE file: {blazefile}') from None

        if smart_blaze:
            # assert 'S2D_A' in s2dfile, 'Not a de-blazed S2D file'
            s2d_blaze_file = s2dfile.replace('S2D_A', 'S2D_BLAZE_A').replace('S2D_TELL_CORR_A', 'S2D_BLAZE_TELL_CORR_A')
            print(f'Using file {s2d_blaze_file} for blaze correction')
            if os.path.exists(s2d_blaze_file):
                with fits.open(s2d_blaze_file) as hdu_s2d_blaze:
                    with np.errstate(invalid='ignore'):
                        blaze = hdu_s2d_blaze[1].data / hdu[1].data 
        else:
            with fits.open(blazefile) as hdu_blaze:
                blaze = hdu_blaze[1].data

    ll = hdu[5].data
    if ll.min() > mask['lambda'].max():
        raise ValueError('No wavelength overlap between S2D and mask lines')
    if ll.max() < mask['lambda'].min():
        raise ValueError('No wavelength overlap between S2D and mask lines')

    # dll used to be stored in a separate file (?), now it's in the S2D
    # dllfile = hdu[0].header['HIERARCH ESO PRO REC1 CAL8 NAME']
    # dllfile = find_file(dllfile, ssh)
    # dll = fits.open(dllfile)[1].data #.astype(np.float64)
    dll = hdu[7].data

    if do_flux_corr:
        # get the flux correction stored in the S2D file
        keyword = 'HIERARCH ESO QC ORDER%d FLUX CORR'
        flux_corr = np.array([hdu[0].header[keyword % o] for o in range(1, norders + 1)])
        # fit a polynomial and evaluate it at each order's wavelength
        # orders with flux_corr = 1 are ignored in the polynomial fit
        fit_nb = (flux_corr != 1.0).sum()
        ignore = norders - fit_nb
        # see espdr_science.c : espdr_correct_flux
        poly_deg = round(8 * fit_nb / norders)
        llc = hdu[5].data[:, order_len // 2]
        coeff = np.polyfit(llc[ignore:], flux_corr[ignore:], poly_deg)
        # corr_model = np.ones_like(hdu[5].data, dtype=np.float32)
        corr_model = np.polyval(coeff, hdu[5].data)
        if verbose:
            print('Performing flux correction', end=' ')
            print(f'(discarding {ignore} orders; '
                  f'polynomial of degree {poly_deg})')
            
            print("Flux correction performed with min/max values:", end=' ')
            print(f'{flux_corr.min():.6f}/{flux_corr.max():.6f}')
    else:
        if verbose:
            print('No flux correction performed')
        corr_model = np.ones(norders)


    kwargs = {}
    kwargs['data'] = [None] + [hdu[i].data for i in range(1, 6)]
    kwargs['dll'] = dll
    kwargs['blaze'] = blaze
    kwargs['corr_model'] = corr_model
    kwargs['rvarray'] = rvarray
    kwargs['mask'] = mask
    kwargs['BERV'] = BERV
    kwargs['BERVMAX'] = BERVMAX
    kwargs['mask_width'] = mask_width
    # kwargs['verbose'] = verbose

    if verbose:
        print('Calculating...', end=' ', flush=True)

    start = pytime.time()

    pool = multiprocessing.Pool(ncores)

    ## progress bar
    ccfs, ccfes, ccfqs = zip(
        *tqdm(pool.imap(_dowork, product(orders, [kwargs, ])),
              total=len(orders))
    )
    ## no progress bar
    # ccfs, ccfes, ccfqs = zip(*pool.map(_dowork, product(orders, [kwargs, ])))
    
    pool.close()

    end = pytime.time()

    if verbose:
        print(f'done in {end - start:.2f} seconds')

    assert np.isfinite(ccfs).all()
    assert np.isfinite(ccfes).all()

    if return_sum:
        # sum the CCFs over the orders
        ccf = np.concatenate([ccfs, np.array(ccfs).sum(axis=0, keepdims=True)])
        # quadratic sum of the errors
        qsum = np.sqrt(np.sum(np.square(ccfes), axis=0))
        ccfe = np.concatenate([ccfes, qsum.reshape(1, -1)])
        # sum the qualities
        ccfq = np.concatenate([ccfqs, np.array(ccfqs).sum(axis=0, keepdims=True)])

        if full_output:
            return ccf, ccfe, ccfq, kwargs
        else:
            return ccf, ccfe, ccfq
    else:
        if full_output:
            return np.array(ccfs), np.array(ccfes), np.array(ccfqs), kwargs
        else:
            return np.array(ccfs), np.array(ccfes), np.array(ccfqs)


def calculate_s1d_ccf_parallel(s1dfile, rvarray, mask_file='ESPRESSO_G2.fits',
                               mask_width=0.5, ncores=None, verbose=True,
                               full_output=False, ignore_blaze=True,
                               skip_flux_corr=False, ssh=None):
    """
    docs
    """
    raise NotImplementedError

    hdu = fits.open(s1dfile)

    # if ncores is None:
    #     ncores = get_ncores()
    ncores = 1  # TODO: make computation parallel for S1D
    print(f'Using {ncores} CPU cores for the calculation')

    BERV = hdu[0].header['HIERARCH ESO QC BERV']
    BERVMAX = hdu[0].header['HIERARCH ESO QC BERVMAX']

    wave = hdu[1].data.wavelength.astype(np.float64)
    wave_air = hdu[1].data.wavelength_air.astype(np.float64)
    flux = hdu[1].data.flux.astype(np.float64)
    error = hdu[1].data.error.astype(np.float64)
    qual = hdu[1].data.quality.astype(int)
    blaze = np.ones_like(wave, dtype=np.float64)

    #? not sure here
    dll = np.diff(wave_air)
    dll = np.r_[dll, dll[-1]]

    ## CCF mask
    mask_file = find_file(mask_file, ssh, verbose)
    mask = fits.open(mask_file)[1].data

    ccf, ccfe, ccfq = espdr_compute_CCF_fast(wave_air, dll, flux, error, blaze, qual, rvarray, mask,
                                             BERV, BERVMAX, mask_width=mask_width)
    kw = None
    return ccf, ccfe, ccfq, kw



def calculate_ccf(filename, mask=None, rvarray=None, **kwargs):
    """
    Calculate the CCF for an S2D file and save the result to a fits file.

    Args:
        filename (str):
            The name of the S2D file
        mask (str, Mask, optional):
            The identifier for the CCF mask to use or an `iCCF.Mask` object. If
            not provided, try to extact it from the S2D file. If a string, a
            file 'INST_mask.fits' should exist (not necessarily in the current
            directory)
        rvarray (array, optional):
            RV array where to calculate the CCF. If not provided, try to extact
            it from the S2D file
        clobber (bool, default True):
            Whether to replace output CCF file even if it exists.
        verbose (bool, default True):
            Print status messages and progress bar.
        **kwargs
            Keyword arguments passed directly to `calculate_s2d_ccf_parallel`
    """

    # read original S2D file
    directory, file = os.path.split(filename)
    s2dhdu_header = fits.getheader(filename)

    instrument = getINSTRUMENT(filename)

    mask_widths = {
        'ESPRESSO': 0.5,
        'HARPS': 0.82,
        'NIRPS': 1.0
    }
    kwargs.setdefault('mask_width', mask_widths[instrument])

    if mask is None:
        mask = s2dhdu_header['HIERARCH ESO QC CCF MASK']
    
    print('Using CCF mask:', mask)
    if isinstance(mask, str):
        mask_str = mask
        mask_file = f"{instrument}_{mask}.fits"
        mask_file = find_file(mask_file)
        with fits.open(mask_file) as hdul:
            mask = hdul[1].data
    elif isinstance(mask, Mask):
        mask_str = f'{mask.mask_name}_{mask.nlines}'
    else:
        raise TypeError('mask must be a string or a `Mask` object')

    kwargs['mask'] = mask

    output = kwargs.pop('output', None)
    if output:
        ccf_file = output
        ccf_file += '' if ccf_file.endswith('.fits') else '.fits'
    else:
        end = f'_CCF_{mask_str}_iCCF.fits'
        ccf_file = os.path.splitext(filename)[0] + end

    clobber = kwargs.pop('clobber', True)
    if os.path.exists(ccf_file) and not clobber:
        if kwargs.get('verbose', True):
            print(f'Output CCF file exists: {ccf_file}')
        return ccf_file

    if rvarray is None:
        try:
            OBJ_RV = s2dhdu_header['HIERARCH ESO OCS OBJ RV']
        except KeyError:
            OBJ_RV = s2dhdu_header['HIERARCH ESO TEL TARG RADVEL']
        start = s2dhdu_header['HIERARCH ESO RV START']
        step = s2dhdu_header['HIERARCH ESO RV STEP']
        end = OBJ_RV + (OBJ_RV - start)
        rvarray = np.arange(start, end + step, step)

    flux_corr = kwargs.get(
        'do_flux_corr', 
        bool(s2dhdu_header.get('HIERARCH ESO QC SCIRED FLUX CORR CHECK'))
    )
    kwargs['do_flux_corr'] = flux_corr

    # if s1d or 'S1D' in filename:
    #     ccf, ccfe, ccfq, kw = calculate_s1d_ccf_parallel(
    #         filename, rvarray, full_output=True, **kwargs)
    # else:
    ccf, ccfe, ccfq, kw = calculate_s2d_ccf_parallel(filename, rvarray, order='all', 
                                                     full_output=True, **kwargs)

    # in the pipeline, data are saved as floats
    ccf = ccf.astype(np.float32)
    ccfe = ccfe.astype(np.float32)
    ccfq = ccfq.astype(np.int32)

    Ind = Indicators(rvarray, ccf[-1], ccfe[-1])

    phdr = fits.Header(s2dhdu_header, copy=True)
    phdr['HIERARCH ESO RV START'] = rvarray[0]
    phdr['HIERARCH ESO RV STEP'] = np.ediff1d(rvarray)[0]
    phdr['HIERARCH ESO QC CCF MASK'] = mask_str

    if np.isnan(Ind.RV):
        warnings.warn('CCF RV is NaN')
    else:
        phdr['HIERARCH ESO QC CCF RV'] = Ind.RV
        phdr['HIERARCH ESO QC CCF RV ERROR'] = Ind.RVerror
        phdr['HIERARCH ESO QC CCF FWHM'] = Ind.FWHM
        phdr['HIERARCH ESO QC CCF FWHM ERROR'] = Ind.FWHMerror
        phdr['HIERARCH ESO QC CCF BIS SPAN'] = Ind.BIS
        phdr['HIERARCH ESO QC CCF BIS SPAN ERROR'] = Ind.BISerror
        phdr['HIERARCH ESO QC CCF CONTRAST'] = Ind.contrast
    # # phdr['HIERARCH ESO QC CCF CONTRAST ERROR'] = Ind.contrasterror # TODO
    # # 'ESO QC CCF FLUX ASYMMETRY' # TODO

    phdu = fits.PrimaryHDU(header=phdr)

    # science data, the actual CCF!
    hdr1 = fits.Header()
    hdr1['EXTNAME'] = 'SCIDATA'
    hdu1 = fits.ImageHDU(ccf, header=hdr1)

    # CCF errors
    hdr2 = fits.Header()
    hdr2['EXTNAME'] = 'ERRDATA'
    hdu2 = fits.ImageHDU(ccfe, header=hdr2)

    # quality flag
    hdr3 = fits.Header()
    hdr3['EXTNAME'] = 'QUALDATA'
    hdu3 = fits.ImageHDU(ccfq, header=hdr3)


    hdul = fits.HDUList([phdu, hdu1, hdu2, hdu3])
    
    if kwargs.get('verbose', True):
        print('Output to:', ccf_file)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning)
        hdul.writeto(ccf_file, overwrite=True, checksum=True)

    return ccf_file






# @njit
# def espdr_compute_CCF_numba_fast(ll: np.ndarray, dll: np.ndarray, flux: np.ndarray, error: np.ndarray,
#                                  blaze: np.ndarray, quality: np.ndarray, RV_table: np.ndarray,
#                                  mask_wave: np.ndarray, mask_contrast: np.ndarray,
#                                  berv: float, bervmax: float, mask_width: float=0.5):
#     c: float = 299792.458
#     nx_s2d = flux.size
#     # ny_s2d = 1  #! since this function computes only one order
#     n_mask = mask_wave.size
#     nx_ccf = len(RV_table)
#     ccf_flux = np.zeros_like(RV_table)
#     ccf_error = np.zeros_like(RV_table)
#     ccf_quality = np.zeros_like(RV_table)
#     dll2 = dll / 2.0  # cpl_image_divide_scalar_create(dll,2.);
#     ll2 = ll - dll2  # cpl_image_subtract_create(ll,dll2);
#     #? this mimics the pipeline (note that cpl_image_get indices start at 1)
#     imin = 1
#     imax = nx_s2d
#     while (imin < nx_s2d and quality[imin - 1] != 0):
#         imin += 1
#     while (imax > 1 and quality[imax - 1] != 0):
#         imax -= 1
#     if imin >= imax:
#         return
#     #? note that cpl_image_get indices start at 1, hence the "-1"s
#     llmin = ll[imin + 1 - 1] / (1 + berv / c) * (1 + bervmax / c) / (1 + RV_table[0] / c)
#     llmax = ll[imax - 1 - 1] / (1 + berv / c) * (1 - bervmax / c) / (1 + RV_table[nx_ccf - 1] / c)
#     imin, imax = 0, n_mask - 1
#     #? turns out cpl_table_get indices start at 0...
#     while (imin < n_mask and mask_wave[imin] < (llmin + 0.5 * mask_width / c * llmin)):
#         imin += 1
#     while (imax >= 0     and mask_wave[imax] > (llmax - 0.5 * mask_width / c * llmax)):
#         imax -= 1
#     for i in range(imin, imax + 1):
#         #? cpl_array_get indices also start at 0
#         llcenter = mask_wave[i] * (1. + RV_table[nx_ccf // 2] / c)
#         index_center = 1
#         while(ll[index_center-1] < llcenter): index_center += 1
#         contrast = mask_contrast[i]
#         w = contrast * contrast
#         for j in range(0, nx_ccf):
#             llcenter = mask_wave[i] * (1. + RV_table[j] / c)
#             llstart = llcenter - 0.5 * mask_width / c * llcenter
#             llstop = llcenter + 0.5 * mask_width / c * llcenter
#             index1 = 1
#             while(ll2[index1-1] < llstart): index1 += 1
#             index2 = index1
#             while (ll2[index2-1] < llcenter): index2 += 1
#             index3 = index2
#             while (ll2[index3-1] < llstop): index3 += 1;
#             k = j
#             for index in range(index1, index3):
#                 ccf_flux[k] += w * flux[index-1] / blaze[index-1] * blaze[index_center-1]
#             ccf_flux[k] += w * flux[index1 - 1 - 1] * (ll2[index1-1] - llstart) / dll[index1 - 1 - 1] / blaze[index1 - 1 - 1] * blaze[index_center - 1]
#             ccf_flux[k] -= w * flux[index3 - 1 - 1] * (ll2[index3-1] - llstop) / dll[index3 - 1 - 1] / blaze[index3 - 1 - 1] * blaze[index_center - 1]
#             ccf_error[k] += w * w * error[index2 - 1 - 1] * error[index2 - 1 - 1]
#             ccf_quality[k] += quality[index2 - 1 - 1]
#     # my_error = cpl_image_power(*CCF_error_RE,0.5);
#     ccf_error = np.sqrt(ccf_error)
#     return ccf_flux, ccf_error, ccf_quality