# -*- coding: utf-8 -*-

import numpy as np
from scipy.fftpack import fft2
from scipy.fftpack import ifft2
from scipy.fftpack import fftshift
import sys

PI     = np.pi
TWO_PI = 2. * PI

class GaussianRandomField(object):
    '''
    Returns a gaussian random field given a function P(k)
    '''
    def __init__(self, pupil, Pk2):

        # Number of samples is consistent with pupil function
        self.samples = pupil.samples

        # Pass in a power-squared expression
        self.Pk2 = Pk2

        kx = np.arange(-self.samples/2, self.samples/2) # Both kx and ky are same

        self.KX, self.KY = np.meshgrid(kx, kx, indexing='xy')

    def Pk(self, kx, ky):
        return np.where(kx**2. + ky**2. > 0., (self.Pk2((kx**2. + ky**2.)**0.5))**0.5, 0.0)

    def randomfield(self):
        noise = fftshift(fft2(np.random.normal(size = (self.samples, self.samples))))
        amplitude = self.Pk(self.KX, self.KY)
        
        amplitude = np.where(amplitude > 1e-15, amplitude, 0.)
        return fftshift(ifft2(noise * amplitude))
