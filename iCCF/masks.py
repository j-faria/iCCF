import os
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits

from .utils import find_data_file, doppler_shift_wave


class Mask:
    def __init__(self, mask, instrument=None):
        if instrument is None:
            instrument = 'ESPRESSO'
        
        self.mask_name = mask
        self.instrument = instrument

        mask_file = self._mask_file = f'{instrument}_{mask}.fits'

        if not os.path.exists(mask_file):
            try:
                mask_file = find_data_file(mask_file)
            except FileNotFoundError as e:
                raise FileNotFoundError(f'Could not find file "{mask_file}"') from None
        
        m = fits.open(mask_file)
        self.wavelength = m[1].data['lambda'].copy()
        self.contrast = m[1].data['contrast'].copy()
        self._wavelength = self.wavelength.copy()
        self._contrast = self.contrast.copy()

    @classmethod
    def from_arrays(cls, wavelength, contrast, name=None, instrument=None):
        self = cls.__new__(cls)
        self.wavelength = wavelength
        self.contrast = contrast
        self._wavelength = self.wavelength.copy()
        self._contrast = self.contrast.copy()
        self.mask_name = name or 'unnamed'
        self.instrument = None
        return self


    @property
    def nlines(self):
        return self.wavelength.size
    
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
        self.wavelength = self._wavelength
        self.contrast = self._contrast

    def plot(self, ax=None, rv=0, down=False, factor=1, show_original=True, norm=False,
             **kwargs):
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