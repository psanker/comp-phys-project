# -*- coding: utf-8 -*-

import numpy as np

from abstractpf import AbstractPupilFunction

class SimplePupilFunction(AbstractPupilFunction):
    '''
    Simplest case of pupil function:
    - P(x, y) = 1 inside and 0 outside of aperture
    - W(x, y) = 1 for no phase shifting
    '''

    def pFunc(self, x, y):
        return np.where(x**2 + y**2 > (self.diameter / 2.)**2, 0., 1.)

    def wFunc(self, r, phi):
        return np.ones((self.samples, self.samples))
