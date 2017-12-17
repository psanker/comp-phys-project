import numpy as np

from .abstractpf import AbstractPupilFunction

class ModelPupilFunction(AbstractPupilFunction):
    '''
    Model pupil function

    This is our **real** model

    - P(x, y) = Same as the SimplePupilFunction, but with a hole with radius 'b'
    - W(x, y) = 0 for no phase shifting
    '''

    strut_width = 0.04

    def pFunc(self, x, y):
        pass1 = np.where((x)**2 + (y)**2 <= (self.radius())**2, 1., 0.)
        pass2 = np.where((x)**2 + (y)**2 > ((self.opts['b'] / 2.)**2), pass1, 0.)
        pass3 = np.where(np.abs(x) < ModelPupilFunction.strut_width*self.radius(), 0., pass2)
        pass4 = np.where(np.abs(y) < ModelPupilFunction.strut_width*self.radius(), 0., pass3)
        return pass4

    def wFunc(self, x, y):
        return np.zeros((self.samples, self.samples))


