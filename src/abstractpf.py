# -*- coding: utf-8 -*-

import numpy as np
from scipy.fftpack import fft2
from scipy.fftpack import fftshift
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
        self.spectrum = TWO_PI / np.linspace(400, 700, self.samples) # k-space of visual spectrum -- units of nm^(-1) !

        # Reset private variables in object
        self._clear()

        # Interpret any options passed to the constructor
        self.applySettings(opts)

    def _clear(self):
        self._X           = None
        self._Y           = None

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
        elif k == 'spectrum':
            self.k = v

    def configurationMesh(self):
        '''
        For FFT padding, the image is defined from [0, 2D] on both x & y domains

        Also, keep in mind the Nyquist freq 1 / 2*T where T is the spacing
        Since 2D = N*T, T = 2D / N
        '''
        if self._X is None  or self._Y is None:
            x = np.linspace(0, 2.*self.diameter, self.samples)
            y = np.linspace(0, 2.*self.diameter, self.samples)

            X, Y = np.meshgrid(x, y, indexing='xy') # Note: this behaves like X[j, i]

            # Store mesh for later use
            self._X = X
            self._Y = Y

        return self._X, self._Y

    def radius(self):
        # Shortcut function to the radius
        return self.diameter / 2.

    def nyqfreq(self):
        # Shortcut to the Nyquist frequency
        return self.samples / (2. * self.diameter) # 1 / (2 * (2D / N))

    def render(self, k):
        '''
        Render the pupil function for the provided spectrum and diameter
        '''
        X, Y = self.configurationMesh()

        return self.pFunc(X,Y) * np.exp(-k*1j * self.wFunc(X, Y))

    def psf(self, k):
        '''
        FFT the pupil function, given its parameters, and produce the PSF
        '''

        shift = fftshift(fft2(self.render(k)))

        return np.abs(np.log(1. + shift))**2.

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
