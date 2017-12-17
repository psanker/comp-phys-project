# -*- coding: utf-8 -*-

import numpy as np
from scipy.fftpack import fft2
from scipy.fftpack import ifft2
from scipy.fftpack import fftshift

PI     = np.pi
TWO_PI = 2. * PI

class GaussianRandomField(object):
    '''
    Returns a gaussian random field given a function P(k)
    '''
    def __init__(self, pupil):

        # Number of samples is consistent with pupil function
        self.samples = pupil.samples

        kx               = np.arange(-self.samples/2, self.samples/2) # Both kx and ky are same
        self.KX, self.KY = np.meshgrid(kx, kx, indexing='xy')

    def randomfield(self, pk):
        noise = fft2(np.random.normal(size=(self.samples, self.samples)))

        pks = np.conj(pk)

        amplitude = np.sqrt(pks * pk)

        return fftshift(ifft2(noise * amplitude))
