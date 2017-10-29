# -*- coding: utf-8 -*-

import numpy as np

from .abstractpf import AbstractPupilFunction

class SimplePupilFunction(AbstractPupilFunction):
    '''
    Simplest case of pupil function:
    - P(x, y) = 1 inside and 0 outside of aperture
    - W(x, y) = 0 for no phase shifting
    '''

    def pFunc(self, x, y):
        return np.where((x - (self.diameter))**2 + (y - (self.diameter))**2 > (self.R)**2, 0., 1.)

    def wFunc(self, r, phi):
        return np.zeros((self.samples, self.samples))
