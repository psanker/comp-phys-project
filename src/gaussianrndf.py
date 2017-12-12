# -*- coding: utf-8 -*-

import numpy as np
from scipy.fftpack import fft2
from scipy.fftpack import ifft2
from scipy.fftpack import fftshift
import sys

PI     = np.pi
TWO_PI = 2. * PI

import numpy as np

TWO_PI = 2. * np.pi

class GaussianRandomField(object):
    '''
    Returns a gaussian random field given a function P(k)
    '''
    def __init__(self, pupil):

        # Number of samples is consistent with pupil function
        self.samples = pupil.samples

        kx = np.arange(-self.samples/2, self.samples/2) # Both kx and ky are same

        self.KX, self.KY = np.meshgrid(kx, kx, indexing='xy')

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

    def randomfield(self, diameter):
        r0 = 20e-2 # coherence
        a  = 0.162

        phase = a * (diameter / r0)**(5/6)

        noise = fftshift(fft2(np.random.normal(size=(self.samples, self.samples), scale=phase)))
        amplitude = self.Pk(self.KX, self.KY) * np.conj(self.Pk(self.KX, self.KY))

        amplitude = np.where(amplitude > 1e-15, amplitude, 0.)
        return fftshift(ifft2(noise * amplitude))
