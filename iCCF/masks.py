import os
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits

from .utils import find_data_file, doppler_shift_wave


class Mask:
    """ Hold information about a CCF mask for a given instrument. """
    def __init__(self, mask, instrument=None):
        """
        Args:
            mask (str):
                Mask name (e.g. G2, M4, ...).
            instrument (str, optional):
                Instrument name (e.g. NIRPS). Defaults to ESPRESSO.

        Attributes:
            wavelength (array):
                Wavelength array in Ängstroms.
            contrast (array):
                Mask contrast array.

        Raises:
            FileNotFoundError: If the mask file cannot be found.
        """

        if mask.endswith('.mas'): # old masks from HARPS
            self.instrument = 'HARPS'
            self.mask_name = mask[:-4]
            mask_file = self._mask_file = mask
            old_format = True
        else:
            self.mask_name = mask
            self.instrument = instrument or 'ESPRESSO'
            mask_file = self._mask_file = f'{self.instrument}_{mask}.fits'
            old_format = False

        if not os.path.exists(mask_file):
            try:
                mask_file = find_data_file(mask_file)
            except FileNotFoundError as e:
                raise FileNotFoundError(f'Could not find file "{mask_file}"') from None

        if old_format:
            data = np.loadtxt(mask_file, unpack=True)
            self.wavelength = 0.5*(data[0] + data[1]).copy()
            self.contrast = data[2].copy()
        else:        
            m = fits.open(mask_file)
            self.wavelength = m[1].data['lambda'].copy()
            self.contrast = m[1].data['contrast'].copy()

        self._wavelength = self.wavelength.copy()
        self._contrast = self.contrast.copy()

    @classmethod
    def from_arrays(cls, wavelength, contrast, name: str = None, instrument: str = None):
        """ Create a mask from wavelength and contrast arrays. """
        self = cls.__new__(cls)
        self.wavelength = wavelength
        self.contrast = contrast
        self._wavelength = self.wavelength.copy()
        self._contrast = self.contrast.copy()
        self.mask_name = name or 'unnamed'
        self.instrument = None
        return self

    @staticmethod
    def available_masks():
        av = find_data_file('NIRPS_*.fits')
        av += find_data_file('ESPRESSO_*.fits')
        av += find_data_file('HARPS_*.fits')
        return [path.stem for path in av]

    @property
    def nlines(self):
        """ Number of lines (currently) in the mask """
        return self.wavelength.size

    @property
    def size(self):
        """ Number of lines (currently) in the mask """
        return self.nlines
    
    def __repr__(self):
        if self.instrument is None:
            return f'Mask({self.mask_name}, {self.nlines} lines)'
        return f'Mask({self.mask_name}, {self.instrument}, {self.nlines} lines)'

    def __getitem__(self, key):
        if key in ('wavelength', 'lambda'):
            return self.wavelength
        elif key in ('contrast',):
            return self.contrast
        else:
            raise KeyError(key)

    def select_wavelength(self, min=None, max=None):
        """ Select lines based on min or max wavelengths """
        λ = self.wavelength
        mask = np.full(λ.size, True)
        if min is not None:
            mask &= λ > min
        if max is not None:
            mask &= λ < max

        if not mask.all():
            print(f'removing {(~mask).sum()} lines')

        self.wavelength = self.wavelength[mask]
        self.contrast = self.contrast[mask]

    def select_contrast(self, min=None, max=None):
        """ Select lines based on min or max contrast """
        c = self._contrast
        mask = np.full(c.size, True)
        if min is not None:
            mask &= c > min
        if max is not None:
            mask &= c < max

        if not mask.all():
            print(f'removing {(~mask).sum()} lines')

        self.wavelength = self._wavelength[mask].copy()
        self.contrast = self._contrast[mask].copy()

    def reset(self):
        """ Reset the mask to the original lines """
        self.wavelength = self._wavelength
        self.contrast = self._contrast

    def plot(self, ax=None, rv=0, down=False, factor=1, show_original=True, norm=False, **kwargs):
        """ Plot the mask

        Args:
            ax (matplotlib.Axes, optional):
                An optional axes to plot on.
            rv (int, optional):
                Radial velocity by which to shift the mask [km/s]. Defaults to 0.
            down (bool, optional):
                Plot mask lines going down. Defaults to False.
            factor (int, optional):
                Factor by which to scale the contrast. Defaults to 1.
            show_original (bool, optional):
                Whether to show the original mask for comparison. Defaults to True.
            norm (bool, optional):
                Normalize the contrast by the maximum value. Defaults to False.

        Returns:
            fig, ax:
                The figure and axes objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        else:
            fig = ax.figure
        
        if down:
            a, b = -self.contrast * factor, 0
            if norm:
                a /= self.contrast.max()
        else:
            a, b = 0, self.contrast * factor
            if norm:
                b /= self.contrast.max()

        w = doppler_shift_wave(self.wavelength, rv)
        ax.vlines(w, a, b, alpha=0.8, **kwargs, 
                  label=self.mask_name)

        if show_original:
            if self.wavelength.size != self._wavelength.size:
                w = doppler_shift_wave(self._wavelength, rv)
                ax.vlines(w, -self._contrast * factor, 0, alpha=0.2, color='k',
                          label='original')

        ax.legend()
        ax.set(xlabel=r'wavelength [$\AA$]', ylabel='contrast')
        return fig, ax


