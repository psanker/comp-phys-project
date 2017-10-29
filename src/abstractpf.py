# -*- coding: utf-8 -*-

import numpy as np
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
        self._pupilRender = None
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
        For FFT padding, the image is defined from [-D, D] on both x & y domains
        '''
        if self._X is None  or self._Y is None:
            x = np.linspace(-self.diameter, self.diameter, self.samples)
            y = np.linspace(-self.diameter, self.diameter, self.samples)

            X, Y = np.mgrid(x, y, indexing='xy') # Note: this behaves like X[j, i]

            # Store mesh for later use
            self._X = X
            self._Y = Y

        return self._X, self._Y

    def render(self, force=False):
        '''
        Render the pupil function for the provided spectrum and diameter
        '''
        if self._pupilRender is None or force:
            r   = np.linspace(0, self.diameter / 2., self.samples)
            phi = np.linspace(0, TWO_PI, self.samples)

            # Define a grid on the optics
            rv, phiv = np.meshgrid(r, phi)

            P = self.pFunc(rv, phiv)
            W = self.wFunc(rv, phiv)

            exponent = -1j * W.dot(self.spectrum)

            rend = P.dot(np.exp(exponent))
            self._pupilRender = rend

        return self._pupilRender

    def transform(self):
        '''
        FFT the pupil function, given its parameters, and produce an image
        '''
        raise Exception('Not implemented yet')

    def pFunc(self, r, phi):
        '''
        The P(r, φ) function for the PSF
        '''
        raise Exception('Not implemented yet')

    def wFunc(self, r, phi):
        '''
        The W(r, φ) function to simulate mirror imperfections
        '''
        raise Exception('Not implemented yet')
