""" Calculate CCFs themselves. So meta! """

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import sys, os
import subprocess
import multiprocessing
from itertools import product
from bisect import bisect_left, bisect_right
from glob import glob
from scipy.interpolate import interp1d
import tqdm


from .utils import doppler_shift_wave



def makeCCF(spec_wave, spec_flux, mask_wave=None, mask_contrast=None,
            mask=None, mask_width=0.82, rvmin=None, rvmax=None, drv=None,
            rvarray=None):
    """
    Cross-correlate an observed spectrum with a mask template.

    For each RV value in rvmin:rvmax:drv (or in rvarray), the wavelength axis of
    the mask is Doppler shifted. The mask is then projected onto the spectrum
    (using the provided `mask_width`) and the sum of the flux that goes through
    the mask "holes" is calculated. This sum is weighted by the mask contrast
    (which corresponds to line depths) to optimally extract the Doppler
    information.

    Parameters
    ----------
    spec_wave : array
        The wavelength of the observed spectrum.
    spec_flux : array
        The flux of the observed spectrum.
    mask_wave : array, optional
        The central wavelength of the mask.
    mask_contrast : array, optional
        The flux (contrast) of the mask.
    mask : array (..., 3), optional
        The mask as an array with lambda1, lambda2, depth.
    mask_width : float, optional [default=0.82]
        Width of the mask "holes", in velocity in km/s.
    rvmin : float, optional
        Minimum radial velocity for which to calculate the CCF [km/s].
    rvmax : float, optional
        Maximum radial velocity for which to calculate the CCF [km/s].
    drv : float, optional
        The radial-velocity step [km/s].
    rvarray : array, optional
        The radial velocities at which to calculate the CCF [km/s]. If this is
        provided, `rvmin`, `rvmax` and `drv` are ignored.

    Returns
    -------
    rv : array The radial-velocity where the CCF was calculated [km/s]. These
        RVs refer to a shift of the mask -- positive values indicate that the
        mask has been red-shifted and negative numbers indicate a blue-shift of
        the mask.
    ccf : array
        The values of the cross-correlation function.
    """
    if rvarray is None:
        if rvmin is None or rvmax is None or drv is None:
            raise ValueError("Provide `rvmin`, `rvmax`, and `drv`.")
        # check order of rvmin and rvmax
        if rvmax <= rvmin:
            raise ValueError("`rvmin` should be smaller than `rvmax`.")
        rvarray = np.arange(rvmin, rvmax + drv / 2, drv)

    wave_resolution = spec_wave[1] - spec_wave[0]

    if mask is None:
        if mask_wave is None:
            raise ValueError("Provide the mask wavelengths in `mask_wave`.")
        if mask_contrast is None:
            raise ValueError(
                "Provide the mask wavelengths in `mask_contrast`.")

        mask = np.c_[doppler_shift_wave(mask_wave, -mask_width / 2),
                     doppler_shift_wave(mask_wave, mask_width /
                                        2), mask_contrast]

    ccfarray = np.zeros_like(rvarray)
    for i, RV in enumerate(rvarray):
        nlines = 0
        CCF = 0.0

        mask_rv_shifted = np.copy(mask)
        mask_rv_shifted[:, :2] = doppler_shift_wave(mask[:, :2], RV)

        # region of intersection between the RV-shifted mask and the spectrum
        region = (spec_wave[0] < mask_rv_shifted[:, 0]) & (
            mask_rv_shifted[:, 1] < spec_wave[-1])
        mask_rv_shifted = mask_rv_shifted[region]

        # for every line in the mask
        for mask_line_start, mask_line_end, mask_line_depth in mask_rv_shifted:

            if mask_line_end + wave_resolution >= spec_wave[-1]:
                break

            # find the limiting indices in spec_wave, corresponding to the start
            # and end wavelength of the mask
            linePixelIni = bisect_left(spec_wave, mask_line_start)
            linePixelEnd = bisect_right(spec_wave, mask_line_end)

            # fraction of the spectrum inside the mask hole at the start
            lineFractionIni = (
                spec_wave[linePixelIni] - mask_line_start) / wave_resolution
            # fraction of the spectrum inside the mask hole at the end
            lineFractionEnd = 1 - abs(
                mask_line_end - spec_wave[linePixelEnd]) / wave_resolution

            CCF += mask_line_depth * np.sum(
                spec_flux[linePixelIni:linePixelEnd])
            CCF += mask_line_depth * lineFractionIni * spec_flux[linePixelIni -
                                                                 1]
            CCF += mask_line_depth * lineFractionEnd * spec_flux[linePixelEnd +
                                                                 1]
            nlines += 1

        ccfarray[i] = CCF

    return rvarray, ccfarray


def espdr_compute_CCF_fast(ll, dll, flux, error, blaze, quality, RV_table,
                           mask, berv, bervmax, mask_width=0.5):
    c = 299792.458

    # nx_s2d = len(flux)
    n_mask = mask['lambda'].size #len(mask)
    # n_mask = len(mask)
    nx_ccf = len(RV_table)

    ccf_flux = np.zeros_like(RV_table)
    ccf_error = np.zeros_like(RV_table)
    ccf_quality = np.zeros_like(RV_table)

    dll2 = dll / 2.0  # cpl_image_divide_scalar_create(dll,2.);
    ll2 = ll - dll2  # cpl_image_subtract_create(ll,dll2);

    imin = np.where(quality == 0)[0][0]
    imax = len(quality) - np.where(quality[::-1] == 0)[0][0] - 1
    # print(imin, imax)

    if imin >= imax:
        return

    llmin = ll[imin + 1] / (1. + berv / c) * (1. + bervmax / c) / (1. + RV_table[0] / c)
    llmax = ll[imax - 1] / (1. + berv / c) * (1. - bervmax / c) / (1. + RV_table[nx_ccf - 1] / c)
    # print(llmin, llmax)

    imin = 0
    imax = n_mask - 1

    while (imin < n_mask
           and mask['lambda'][imin] < (llmin + 0.5 * mask_width / c * llmin)):
        imin += 1
    while (imax >= 0
           and mask['lambda'][imax] > (llmax - 0.5 * mask_width / c * llmax)):
        imax -= 1
    # print(imin, imax)

    for i in range(imin, imax + 1):
        llcenter = mask['lambda'][i] * (1. + RV_table[nx_ccf // 2] / c)

        index_center = np.where(ll < llcenter)[0][-1] + 1

        contrast = mask['contrast'][i]
        w = contrast * contrast

        for j in range(0, nx_ccf):
            llcenter = mask['lambda'][i] * (1. + RV_table[j] / c)
            llstart = llcenter - 0.5 * mask_width / c * llcenter
            llstop = llcenter + 0.5 * mask_width / c * llcenter

            index1 = np.where(ll2 < llstart)[0][-1] + 1
            # index2 = index1

            index2 = np.where(ll2 < llcenter)[0][-1] + 1
            # index3 = index2

            index3 = np.where(ll2 < llstop)[0][-1] + 1

            k = j

            for index in range(index1, index3):
                ccf_flux[k] += w * flux[index] / blaze[index] * blaze[index_center]

            ccf_flux[k] += w * flux[index1 - 1] * (ll2[index1] - llstart) / dll[index1 - 1] / blaze[index1 - 1] * blaze[index_center]
            ccf_flux[k] -= w * flux[index3 - 1] * (ll2[index3] - llstop) / dll[index3 - 1] / blaze[index3 - 1] * blaze[index_center]

            ccf_error[k] += w * w * error[index2 - 1] * error[index2 - 1]

            ccf_quality[k] += quality[index2 - 1]

    # my_error = cpl_image_power(*CCF_error_RE,0.5);
    ccf_error = np.sqrt(ccf_error)

    return ccf_flux, ccf_error, ccf_quality


def find_dll(s2dfile):
    hdu = fits.open(s2dfile)
    dllfile = hdu[0].header['HIERARCH ESO PRO REC1 CAL7 NAME']
    if os.path.exists(dllfile):
        return dllfile
    elif len(glob(dllfile + '*')) > 1:
        return glob(dllfile + '*')[0]
    else:
        date = hdu[0].header['DATE-OBS']
        

def calculate_s2d_ccf(s2dfile, rvarray, order='all', maskfile='ESPRESSO_G2.fits', mask=None, mask_width=0.5):

    hdu = fits.open(s2dfile)

    if order == 'all':
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
        mask = fits.open(maskfile)[1].data
    else:
        assert 'lambda' in mask, 'mask must contain the "lambda" key'
        assert 'contrast' in mask, 'mask must contain the "contrast" key'
        # s,e,c = np.loadtxt('/home/jfaria/Work/hardrs_3.8/src/config/G2.mas').T
        # mask = {'lambda':0.5*(s+e), 'contrast':c}
        # mask['lambda'] = mask['lambda'][mask['lambda'] > 3780]


    # get the flux correction stored in the S2D file
    keyword = 'HIERARCH ESO QC ORDER%d FLUX CORR'
    flux_corr = [hdu[0].header[keyword % (o + 1)] for o in range(170)]

    ccfs, ccfes = [], []
    for order in orders:
        # WAVEDATA_AIR_BARY
        ll = hdu[5].data[order, :]
        # mean w
        llc = np.mean(hdu[5].data, axis=1)

        dll = fits.open(dllfile)[1].data[order, :]
        # dll = doppler_shift_wave(dll, -BERV, f=1.+1.55e-8)

        # fit an 8th degree polynomial to the flux correction
        corr_model = np.polyval(np.polyfit(llc, flux_corr, 7), llc)

        flux = hdu[1].data[order, :]
        error = hdu[2].data[order, :]
        quality = hdu[3].data[order, :]

        blaze = fits.open(blazefile)[1].data[order, :]

        y = flux * blaze / corr_model[order]
        # y = np.loadtxt('flux_in_pipeline_order0.txt')
        ye = error * blaze / corr_model[order]

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


def find_file(file):
    print('Looking for file:', file)
    # first try here:
    if os.path.exists(file):
        print('\tfound it in current directory')
        return file

    try:
        found = subprocess.check_output(f'locate {file}'.split())
        found = found.decode().split()
        print('\tfound file:', found[-1])
        return found[-1]
    except:
        raise FileNotFoundError(file)


def dowork(args):
    order, kwargs = args
    data = kwargs['data']
    dllfile = kwargs['dllfile']
    blazefile = kwargs['blazefile']
    flux_corr = kwargs['flux_corr']
    rvarray = kwargs['rvarray']
    mask = kwargs['mask']
    BERV = kwargs['BERV']
    BERVMAX = kwargs['BERVMAX']
    mask_width = kwargs['mask_width']
    # verbose = kwargs['verbose']

    # WAVEDATA_AIR_BARY
    ll = data[5][order, :]
    # mean w
    llc = np.mean(data[5], axis=1)

    dll = fits.open(dllfile)[1].data[order, :]

    # fit an 8th degree polynomial to the flux correction
    corr_model = np.polyval(np.polyfit(llc, flux_corr, 7), llc)

    flux = data[1][order, :]
    error = data[2][order, :]
    quality = data[3][order, :]

    blaze = fits.open(blazefile)[1].data[order, :]

    y = flux * blaze / corr_model[order]
    ye = error * blaze / corr_model[order]
    # if verbose:
    #     print('calculating ccf (order %d)...' % order)
    ccf, ccfe, _ = espdr_compute_CCF_fast(ll, dll, y, ye, blaze, quality,
                                            rvarray, mask, BERV, BERVMAX,
                                            mask_width=mask_width)
    return ccf, ccfe


def calculate_s2d_ccf_parallel(s2dfile, rvarray, order='all', maskfile='ESPRESSO_G2.fits', mask_width=0.5, ncores=None, verbose=True, full_output=False):
    hdu = fits.open(s2dfile)

    if ncores is None:
        ncores = len(os.sched_getaffinity(0))

    if order == 'all':
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
    dllfile = find_file(dllfile)
    blazefile = find_file(blazefile)
    
    # CCF mask
    maskfile = find_file(maskfile)
    mask = fits.open(maskfile)[1].data
    
    # get the flux correction stored in the S2D file
    keyword = 'HIERARCH ESO QC ORDER%d FLUX CORR'
    flux_corr = [hdu[0].header[keyword % (o + 1)] for o in range(170)]

    kwargs = {}
    kwargs['data'] = [None] + [hdu[i].data for i in range(1,6)] 
    kwargs['dllfile'] = dllfile
    kwargs['blazefile'] = blazefile
    kwargs['flux_corr'] = flux_corr
    kwargs['rvarray'] = rvarray
    kwargs['mask'] = mask
    kwargs['BERV'] = BERV
    kwargs['BERVMAX'] = BERVMAX
    kwargs['mask_width'] = mask_width
    # kwargs['verbose'] = verbose

    # return kwargs
    # print(list(product(orders, [kwargs,])))
    # return 

    pool = multiprocessing.Pool(ncores)
    if verbose:
        ccfs, ccfes = zip(*tqdm.tqdm(pool.imap_unordered(dowork, product(orders, [kwargs,])), total=len(orders)))
    else:
        ccfs, ccfes = zip(*pool.map(dowork, product(orders, [kwargs,])))
    pool.close()

    if return_sum:
        ccf = np.concatenate([ccfs, np.array(ccfs).sum(axis=0, keepdims=True)])
        ccfe = np.concatenate([ccfes, np.zeros(len(rvarray)).reshape(1, -1)])
        # what to do with the errors?
        if full_output:
            return ccf, ccfe, kwargs
        else:
            return ccf, ccfe
    else:
        if full_output:
            return np.array(ccfs), np.array(ccfes), kwargs
        else:
            return np.array(ccfs), np.array(ccfes)



def calculate_ccf(s2dfile, **kwargs):
    mask = kwargs.pop('mask')
    maskfile = f"ESPRESSO_{mask}.fits"
    kwargs['maskfile'] = maskfile

    ccf, ccfe, kw = calculate_s2d_ccf_parallel(s2dfile, full_output=True, 
                                               **kwargs)

    # read original S2D file
    s2dhdu = fits.open(s2dfile)


    s2dfile = os.path.basename(s2dfile)
    ccf_file = s2dfile[:s2dfile.index('_')] + f'_CCF_{mask}_iCCF.fits'
    rvarray = kwargs['rvarray']

    phdr = fits.Header()
    phdr['HIERARCH ESO RV START'] = rvarray[0]
    phdr['HIERARCH ESO RV STEP'] = np.ediff1d(rvarray)[0]
    phdr['HIERARCH ESO QC BJD'] = s2dhdu[0].header['ESO QC BJD']
    phdr['HIERARCH ESO QC BERV'] = kw['BERV']
    phdr['HIERARCH ESO QC BERVMAX'] = kw['BERVMAX']
    phdr['HIERARCH ESO QC CCF MASK'] = mask
    # at least these keywords should be added
    # 'ESO PRO SCIENCE'
    # 'ESO PRO TECH'
    # 'ESO PRO TYPE'
    # 'ESO QC BJD'
    # 'ESO QC CCF CONTRAST'
    # 'ESO QC CCF CONTRAST ERROR'
    # 'ESO QC CCF FLUX ASYMMETRY'
    # 'ESO QC CCF FWHM'
    # 'ESO QC CCF FWHM ERROR'
    # 'ESO QC CCF RV'
    # 'ESO QC CCF RV ERROR'
    phdu = fits.PrimaryHDU(header=phdr)

    # science data, the actual CCF!
    hdr1 = fits.Header()
    hdr1['EXTNAME'] = 'SCIDATA'
    hdu1 = fits.ImageHDU(ccf, header=hdr1)

    # CCF errors (TO DO)
    hdr2 = fits.Header()
    hdr2['EXTNAME'] = 'ERRDATA'
    hdu2 = fits.ImageHDU(ccfe, header=hdr2)

    # quality flag (TO DO)
    hdr3 = fits.Header()
    hdr3['EXTNAME'] = 'ERRDATA'
    hdu3 = fits.ImageHDU(ccfe, header=hdr3)


    hdul = fits.HDUList([phdu, hdu1, hdu2])
    print('Output to:', ccf_file)
    hdul.writeto(ccf_file, overwrite=True, checksum=True)