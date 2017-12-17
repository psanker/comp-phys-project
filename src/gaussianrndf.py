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

    def fftIndgen(self, n):
        a = np.arange(0, n//2)
        b = np.arange(1, n//2+1)

        b = [-i for i in b[::-1]]
        return a + b

    def Pk(self, kx, ky):
        # http://community.dur.ac.uk/james.osborn/thesis/thesisse3.html
        l_min = 1e-3  # meters
        l_max = 1e2   # meters
        r0    = 20e-2 # meters

        k_max = 5.92 / l_min
        k_min = TWO_PI / l_max

        k2 = kx**2. + ky**2.

        k2 = np.where(k2 < k_min**2, 0., k2)
        k2 = np.where(k2 > k_max**2, 0., k2)

        a  = 0.023

        num = (np.abs(k2 + k_min**2))**(-11./6.)
        dem = r0**(5./3.)
        expfac = np.exp(-1. * (k2 / k_max**2))

        return a * (num / dem) * expfac

    def kolmogorov_Pk(self, kx, ky):
        a  = 0.0023
        r0 = 20e-2 # meters

        k = np.sqrt(kx**2 + ky**2)

        return a * (r0**(-5./3.)) * (k**(-11./3.))

    def randomfield(self):
        noise = fft2(np.random.normal(size=(self.samples, self.samples)))

        pk  = self.Pk(self.KX, self.KY)
        pks = np.conj(pk)

        amplitude = np.sqrt(pks * pk)

        return fftshift(ifft2(amplitude))

    def randomfield2(self):
        noise = fft2(np.random.normal(size=(self.samples, self.samples)))
        amplitude = np.zeros((self.samples, self.samples))

        for i, kx in enumerate(self.fftIndgen(self.samples)):
            for j, ky in enumerate(self.fftIndgen(self.samples)):
                amplitude[i, j] = self.Pk(kx, ky)
                amplitude[i, j] = np.sqrt(np.conj(amplitude[i, j]) * amplitude[i, j])

        return fftshift(ifft2(noise))
