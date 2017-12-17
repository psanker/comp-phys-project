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

        datapk  = None

        if callable(pk):
            # If the passed 'pk' is a function, use the builtin mesh to compute the power spectrum
            datapk = pk(self.KX, self.KY)
        else:
            # Assuming the number crunching has already been done
            datapk = pk

        datapks   = np.conj(datapk)
        amplitude = np.sqrt(datapks * datapk)

        return fftshift(ifft2(noise * amplitude))
