# -*- coding: utf-8 -*-

import numpy as np
from scipy.fftpack import fft2, fftshift, fftfreq
from scipy.ndimage.filters import gaussian_filter
import sys

PI     = np.pi
TWO_PI = 2. * PI

class AbstractPupilFunction(object):
    '''
    An interface for the many implementations that we will need for this project
    Note: The pupil function is defined in configuration space

    Base units for length are in **centimeters**
    '''

    def __init__(self, **opts):

        # Assign defaults for all children
        self.diameter = 2.                                           # Diameter of optics
        self.samples  = 100                                          # Number of sample points per dimension
        self.padscale = 1.                                           # Number of diameters to use as 0-padding
        self.spectrum = TWO_PI / np.linspace(400, 700, self.samples) # k-space of visual spectrum -- units of nm^(-1) !
        self.struts   = 0                                            # Number of struts in the pupil

        # Reset private variables in object
        self._clear()

        # Interpret any options passed to the constructor
        self.applySettings(opts)

    def _clear(self):
        self.opts = {}
        self._X   = None
        self._Y   = None
        self._KX  = None
        self._KY  = None

    def applySettings(self, vals):
        '''
        Apply settings for runtime and object initialization
        '''
        if sys.version_info < (3, 0): # Legacy support
            for k, v in vals.iteritems():
                self.applySetting(k, v)
        else:
            for k, v in vals.items():
                self.applySetting(k, v)

    def applySetting(self, k, v):
        if k == 'diameter':
            if v < 0: # Idiot check
                v *= -1.
            self.diameter = v
        elif k == 'samples':
            self.samples = v
        elif k == 'padscale':
            self.padscale = v
        elif k == 'struts':
            self.struts = v
        else:
            self.opts[k] = v

    def configurationMesh(self):
        '''
        For FFT padding, the image is defined from [-sD, sD] on both x & y domains

        Also, keep in mind the Nyquist freq 1 / 2*T where T is the spacing
        Since 2D = N*T, T = 2D / N
        '''
        if (self._X is None  or self._Y is None):
            scale = self.padscale * self.diameter

            x = np.linspace(-scale, scale, self.samples) # Both x & y are same

            X, Y = np.meshgrid(x, x, indexing='xy') # Note: this behaves like X[j, i]

            # Store mesh for later use
            self._X = X
            self._Y = Y

        return self._X, self._Y

    def fourierMesh(self):
        '''
        Generates the mesh in Fourier space associated with this pupil (i.e. where the PSF lives)
        '''
        if (self._KX is None  or self._KY is None):
            spacing = 1. / (2. * self.nyqfreq())                      # Spacing from Nyquist frequency
            k       = fftshift(fftfreq(self.samples, d=spacing))

            KX, KY  = np.meshgrid(k, k)

            self._KX = KX
            self._KY = KY

        return self._KX, self._KY

    def radius(self):
        # Shortcut function to the radius
        return self.diameter / 2.

    def nyqfreq(self):
        # Shortcut to the Nyquist frequency
        return self.samples / (4. * self.diameter * self.padscale) # 1 / (2 * (2sD / N))

    def render(self, k, filtering=False):
        '''
        Render the pupil function for the provided spectrum and diameter
        '''
        X, Y = self.configurationMesh()

        img = self.pFunc(X, Y) * np.exp(-k*1j * self.wFunc(X, Y))

        if not filtering:
            return img
        else:
            # Use Gaussian filtering on image to smooth edges
            filtered_re = gaussian_filter(img.real, 1., order=0, mode='constant')
            filtered_im = gaussian_filter(img.imag, 1., order=0, mode='constant')
            return filtered_re + 1j*filtered_im

    def psf(self, k=20., filtering=False, noshift=False):
        '''
        FFT the pupil function, given its parameters, and produce the PSF
        '''
        shift_test = self.render(k, filtering=filtering)
        transform  = None

        if noshift:
            transform = fft2(shift_test)
        else:
            transform = fftshift(fft2(shift_test))

        transform *= np.conj(transform)

        return np.abs(np.log10(1. + np.sqrt(transform)))**2.

    def pFunc(self, x, y):
        '''
        The P(x, y) function for the PSF
        '''
        raise Exception('Not implemented yet')

    def wFunc(self, x, y):
        '''
        The W(x, y) function to simulate mirror imperfections
        '''
        raise Exception('Not implemented yet')
