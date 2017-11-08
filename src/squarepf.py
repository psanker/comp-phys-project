import numpy as np

from scipy.fftpack import fft2
from scipy.fftpack import fftshift

from .abstractpf import AbstractPupilFunction

class SquarePupilFunction(AbstractPupilFunction):

    def pFunc(self, x, y):
        step1 = np.where(np.abs(x) <= self.radius(), 1., 0.)
        return np.where(np.abs(y) <= self.radius(), step1, 0.)

    def wFunc(self, x, y):
        return np.zeros((self.samples, self.samples))
