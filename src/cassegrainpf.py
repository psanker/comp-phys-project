import numpy as np

from .abstractpf import AbstractPupilFunction

class CassegrainPupilFunction(AbstractPupilFunction):
    '''
    Cassegrain pupil function
    - P(x, y) = Same as the SimplePupilFunction, but with a hole with radius 'b'
    - W(x, y) = 0 for no phase shifting
    '''

    def pFunc(self, x, y):
        pass1 = np.where((x)**2 + (y)**2 <= (self.radius())**2, 1., 0.)
        pass2 = np.where((x)**2 + (y)**2 > (self.opts['b']**2), pass1, 0.)
        return pass2

    def wFunc(self, x, y):
        return np.zeros((self.samples, self.samples))
